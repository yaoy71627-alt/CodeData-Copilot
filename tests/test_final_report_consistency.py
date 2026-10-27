"""One finalized report drives UI, Markdown and PDF conclusions and patches."""
import copy
import html
import json
import re
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import pytest

import app
from code_analyzer import analyze_code_with_ast
from data_flow_analyzer import analyze_python_data_flow
from consistency_validator import (
    build_analysis_facts, conclusion_summary_lines, finding_summary_line,
    is_final_report, validate_report_consistency,
)
from report import build_report_data, build_markdown_report, build_pdf_report
from repair_validator import validate_repair_suggestions
from patch_generator import build_unified_patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("train.csv")
X = df[["Age", "Fare"]]
y = df["Survived"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
'''


@pytest.fixture
def state(tmp_path):
    source = tmp_path / "titanic.py"; source.write_text(SOURCE, encoding="utf-8")
    csv = tmp_path / "train.csv"
    pd.DataFrame({"Age": [20, 30, None, 50, 40, 25], "Fare": [10, 20, 30, 40, 50, 60],
                  "Cabin": [None] * 6, "Survived": [0, 1, 0, 1, 0, 1]}).to_csv(csv, index=False)
    state = SimpleNamespace()
    with patch.object(app.st, "session_state", state):
        app.analyze_project_files([("titanic.py", source), ("train.csv", csv)], generate_ai=False)
    return state


def pdf_paragraphs(report):
    from reportlab.platypus import Paragraph
    captured = []
    def capture(text, *args, **kwargs):
        captured.append(html.unescape(re.sub(r"<[^>]+>", "", text)))
        return Paragraph(text, *args, **kwargs)
    with patch("reportlab.platypus.Paragraph", side_effect=capture):
        result = build_pdf_report(report)
    assert result.startswith(b"%PDF")
    return captured


def test_final_model_classification_is_exclusive_and_single_source(state):
    report = state.report_data
    assert is_final_report(report)
    c = report["conclusions"]["counts"]
    assert c["total"] == c["confirmed_error"] + c["static_risk"] + c["needs_review"]
    assert c["total"] == report["review_coverage"]["total"]
    assert report["view"]["patch_text"] == report["patch_text"] == state.patch_text
    assert report["view"]["repair_suggestions"] == report["repair_suggestions"] == state.repair_suggestions


def test_all_core_finding_fields_identical_in_projection(state):
    report = state.report_data
    rows = {row["id"]: row for row in report["conclusions"]["finding_rows"]}
    for finding in report["findings"]:
        row = rows[finding["id"]]
        for key in ("id", "type", "evidence", "severity", "confidence", "verification_status"):
            assert row[key] == finding[key]
        for key in ("file", "cell", "line", "column"):
            assert row[key] == finding["location"][key]


def test_markdown_and_pdf_share_exact_conclusion_lines_and_finding_metadata(state):
    report = state.report_data
    md = build_markdown_report(report)
    paragraphs = pdf_paragraphs(report)
    for line in conclusion_summary_lines(report):
        assert line in md
        assert line in paragraphs
    for issue in report["conclusions"]["finding_rows"]:
        line = finding_summary_line(issue)
        assert line in md and line in paragraphs


def test_no_ai_static_exports_remain_complete_and_do_not_claim_runtime(state):
    assert state.agent_report is None
    md = build_markdown_report(state.report_data)
    assert "未生成可校验的结构化 AI 解释" in md
    assert "完整静态诊断" in md
    assert all(label in md for label in (
        "已证实缺陷", "静态风险", "待核实问题", "信息性观察",
    ))
    assert "待人工复核：" not in md
    assert "本次未启用 AI Review" in md
    assert not state.report_data["patch_text"]
    assert state.report_pdf.startswith(b"%PDF")


def test_final_report_is_idempotent_and_does_not_mutate_analyzer_inputs(state):
    snapshot = copy.deepcopy(state.code_result)
    report = copy.deepcopy(state.report_data)
    checked = validate_report_consistency(report)
    assert checked == report and is_final_report(checked)
    assert state.code_result == snapshot
    assert checked is not report


def test_presentation_tampering_is_rebuilt_from_facts(state):
    report = copy.deepcopy(state.report_data)
    report["conclusions"]["counts"]["total"] = 9999
    report["code_quality"]["issues"][0]["severity"] = "Invented"
    checked = validate_report_consistency(report)
    assert checked["conclusions"]["counts"]["total"] != 9999
    assert all(i["severity"] != "Invented" for i in checked["code_quality"]["issues"])
    assert is_final_report(checked)


def test_dataset_numbers_prefer_verified_facts_over_render_cache(state):
    report = copy.deepcopy(state.report_data)
    report["data_quality"]["datasets"][0]["shape"]["rows"] = 9999
    checked = validate_report_consistency(report)
    assert checked["data_quality"]["datasets"][0]["shape"]["rows"] == 6
    assert checked["view"]["data_results"][0]["result"]["shape"]["rows"] == 6
    assert "9999 行" not in build_markdown_report(checked)


def test_unstructured_ai_cannot_override_final_statistics(state):
    report = build_report_data(state.project_overview, state.code_result, state.data_results,
                              "已确定9999个错误；性能严重高估。", [], "fake patch", state.health,
                              analysis_facts=state.analysis_facts)
    md = build_markdown_report(report)
    assert "9999" not in md and "性能严重高估" not in md
    assert "完整静态诊断" in md
    assert report["review_coverage"]["ai_explained"] == 0
    assert not report["patch_text"]


def test_final_repair_downgrade_removes_patch_in_every_projection():
    source = "x = 1\nvalue = x + 1\nprint(value)\n"
    flow = analyze_python_data_flow(source, "demo.py")
    suggestion = {"file": "demo.py", "location": "demo.py", "line": 2,
                  "before_code": "value = x + 1", "after_code": "value = x + 2"}
    repaired = validate_repair_suggestions([suggestion], flow, [])
    assert build_unified_patch(repaired)
    facts = build_analysis_facts([], [], flow)
    report = build_report_data({"project_name": "demo"}, [], [], None, repaired,
                              build_unified_patch(repaired), {}, analysis_facts=facts)
    assert report["patch_text"]
    # Second final validation now sees an undefined variable at the real location.
    report["repair_suggestions"][0]["after_code"] = "value = unknown + 2"
    report["repair_suggestions"][0]["corrected_code"] = "value = unknown + 2"
    checked = validate_report_consistency(report)
    assert checked["repair_suggestions"][0]["validation_status"] == "Failed Validation"
    assert not checked["patch_text"] and not checked["view"]["patch_text"]
    assert not checked["conclusions"]["patch_available"]
    assert "+++ b/demo.py" not in build_markdown_report(checked)
    assert not any("+++ b/demo.py" in p for p in pdf_paragraphs(checked))


def test_contradictory_absence_finding_removed_before_ai_counts_and_render():
    raw = [{"file": "demo.py", "issues": [{"rule_id": "CODE_NO_VALIDATION_SPLIT", "type": "No Explicit Validation Split", "severity": "Medium", "line": 1}]}]
    facts = build_analysis_facts(raw, [], {"split": {"detected": True}})
    report = build_report_data({}, raw, [], None, [], "", {}, analysis_facts=facts)
    assert not report["findings"] and not report["code_quality"]["issues"]
    assert report["conclusions"]["counts"]["total"] == report["review_coverage"]["total"] == 0
    assert not report["view"]["code_result"]


@pytest.mark.parametrize("page", ["Code Analysis", "Data Quality", "AI Review", "Repair", "Report"])
def test_every_page_uses_final_object_instead_of_corrupted_raw_session(state, page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    for key, value in vars(state).items():
        at.session_state[key] = value
    at.session_state["active_page"] = page
    at.session_state["code_result"] = [{"file": "fake.py", "issues": [{"type": "Fake Risk", "severity": "Invented"}]}]
    at.session_state["data_results"] = []
    at.session_state["repair_suggestions"] = []
    at.session_state["patch_text"] = "fake cached patch"
    at.run()
    assert not at.exception
    captions = [item.value for item in at.caption]
    for line in conclusion_summary_lines(state.report_data):
        assert line in captions
    assert not any("Fake Risk" in str(item.value) for item in at.markdown)
    if page == "Repair":
        assert not any("缺少真实原始代码片段" in w.value for w in at.warning)
        assert any(state.report_data["conclusions"]["patch_status"] in w.value for w in at.warning)
    if page == "AI Review":
        metric = next(m for m in at.metric if m.label == "Detected problems")
        assert metric.value == str(state.report_data["conclusions"]["counts"]["total"])
    if page == "Code Analysis":
        table = next(d.value for d in at.dataframe if "ID" in d.value.columns)
        expected = {i["display_id"] for i in state.report_data["code_quality"]["issues"]}
        assert set(table["ID"]) == expected


def test_pdf_canonical_rows_do_not_lose_notebook_cell_position():
    code = [{"file": "demo.ipynb", "cell": 3, "issues": [{"rule_id": "CODE_RISK", "type": "Example Risk", "severity": "High", "confidence": "High", "line": 2, "evidence": "Actual evidence"}]}]
    report = build_report_data({}, code, [], None, [], "", {})
    line = finding_summary_line(report["conclusions"]["finding_rows"][0])
    assert "Cell 3" in line and "Line 2" in line
    assert line in pdf_paragraphs(report)


def test_empty_project_report_has_readable_static_summary():
    report = build_report_data({}, [], [], None, [], "", {})
    assert report["conclusions"]["counts"]["total"] == 0
    assert "检测问题总数：0" in build_markdown_report(report)
    assert build_pdf_report(report).startswith(b"%PDF")


def test_serialized_report_keeps_verified_ai_counts_on_export(state):
    from consistency_validator import render_structured_review
    findings = state.report_data["findings"]
    payload = {"explanations": [{"finding_id": f["id"], "cause": "请核查对应证据的触发条件。",
               "potential_impact": "实际影响需要通过运行对照确认。", "recommendation": "结合执行路径复核。"} for f in findings]}
    review = render_structured_review(payload, state.analysis_facts)
    report = build_report_data(state.project_overview, state.code_result, state.data_results, review,
                              [], "", state.health, analysis_facts=state.analysis_facts)
    restored = json.loads(json.dumps(report, ensure_ascii=False))
    assert is_final_report(restored)
    md = build_markdown_report(restored)
    expected = report["review_coverage"]["ai_explained"]
    assert f"AI 已解释数量：{expected}" in md
    assert restored["review_coverage"]["ai_explained"] == expected
    # If later modified, a real second validation also restores typed annotations.
    restored["patch_text"] = "untrusted diff"
    checked = validate_report_consistency(restored)
    assert checked["review_coverage"]["ai_explained"] == expected
    assert not checked["patch_text"]


def test_cached_profile_and_quality_score_are_reset_from_analysis_snapshot(state):
    report = copy.deepcopy(state.report_data)
    report["data_quality"]["datasets"][0]["quality_score"] = 9999
    report["project_overview"]["file_count"] = 9999
    checked = validate_report_consistency(report)
    assert checked["data_quality"]["datasets"][0]["quality_score"] != 9999
    assert checked["project_overview"]["file_count"] == 2


def test_publish_final_report_synchronizes_all_patch_export_caches(state):
    report = copy.deepcopy(state.report_data)
    report["patch_text"] = "old unverified patch"
    state.patch_text = "old unverified patch"
    state.report_markdown = "old unverified patch"
    state.report_export_fingerprint = None
    with patch.object(app.st, "session_state", state):
        app.publish_final_report(report)
    assert state.patch_text == state.report_data["patch_text"] == ""
    assert "old unverified patch" not in state.report_markdown
    assert state.report_export_fingerprint == state.report_data["final_fingerprint"]
    assert state.report_pdf.startswith(b"%PDF")


def test_pdf_embeds_chinese_font_instead_of_relying_on_viewer_fonts(state):
    pdf = build_pdf_report(state.report_data)
    assert b"/FontFile2" in pdf and b"NotoSansSC" in pdf
    assert (ROOT / "assets/fonts/OFL.txt").is_file()


def test_public_result_layers_do_not_expose_the_legacy_review_count(state):
    report = state.report_data
    counts = report["conclusions"]["counts"]
    md = build_markdown_report(report)
    assert "待人工复核：" not in md
    assert f"需处理问题：{counts['actionable_total']}" in md
    assert f"信息性观察：{counts['informational_observation']}" in md
    assert f"仍需验证或补齐解释数量：{counts['verification_pending']}" in md
    assert counts['verification_pending'] >= counts['needs_review']


def test_home_counts_and_patch_status_ignore_stale_raw_statistics(state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    for key, value in vars(state).items():
        at.session_state[key] = value
    at.session_state["active_page"] = "Home"
    at.session_state["project_stats"] = {"files": 9999, "code_files": 9999, "data_files": 9999, "documentation_files": 0}
    at.run()
    assert not at.exception
    text = "\n".join(str(m.value) for m in at.markdown)
    assert "9999" not in text
    assert "Patch status" in text and "Not generated" in text
