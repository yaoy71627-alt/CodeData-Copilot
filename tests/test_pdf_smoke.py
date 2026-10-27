"""PDF resource, Unicode, multipage and original-case offline product smoke."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from pypdf import PdfReader

import agent
import app
import report
from consistency_validator import conclusion_summary_lines, finding_summary_line


def compact(text):
    return "".join(text.split())


def assert_pdf_conclusions(pdf, final):
    assert pdf.startswith(b"%PDF") and b"%%EOF" in pdf[-30:]
    reader = PdfReader(BytesIO(pdf))
    text = compact("\n".join(page.extract_text() for page in reader.pages))
    for line in conclusion_summary_lines(final):
        assert compact(line) in text
    for row in final["conclusions"]["finding_rows"]:
        assert compact(finding_summary_line(row)) in text
    return reader, text


def sample_report():
    return report.build_report_data({"project_name": "中文项目：数据质量与代码审查"},
        [{"file": "example.ipynb", "cell": 3, "issues": [
            {"rule_id": f"CODE_REVIEW_{i}", "type": "静态风险需要复核", "severity": "High",
             "confidence": "Medium", "line": i + 1, "evidence": "真实静态证据；未执行模型训练或性能对照。"}
            for i in range(30)]}], [], None, [], "", {})


def test_bundled_licensed_font_is_available_at_startup():
    status = report.pdf_font_status()
    assert status["status"] == "embedded"
    root = Path(report.__file__).parent / "assets/fonts"
    assert (root / "OFL.txt").is_file() and (root / "FONT_SOURCE.txt").is_file()


def test_chinese_title_body_and_multipage_pdf():
    final = sample_report()
    final["title"] = "中文项目质量报告"
    pdf = report.build_pdf_report(final)
    reader, text = assert_pdf_conclusions(pdf, final)
    assert len(reader.pages) >= 2
    assert "中文项目质量报告" in text and "真实静态证据" in text
    assert b"/FontFile2" in pdf


def test_missing_ttf_does_not_emit_viewer_dependent_chinese_boxes(monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == "NotoSansSC-Regular.ttf" else original(p))
    assert report.pdf_font_status()["status"] == "unavailable"
    final = sample_report()
    with pytest.raises(RuntimeError, match="CODEDATA_PDF_FONT"):
        report.build_pdf_report(final)
    assert "静态风险需要复核" in report.build_markdown_report(final)


def test_explicit_embedded_cjk_font_works_when_default_path_is_missing(tmp_path, monkeypatch):
    source = Path(report.__file__).parent / "assets/fonts/NotoSansSC-Regular.ttf"
    custom = tmp_path / "custom-chinese.ttf"
    custom.write_bytes(source.read_bytes())
    monkeypatch.setenv("CODEDATA_PDF_FONT", str(custom))
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == "NotoSansSC-Regular.ttf" else original(p))
    assert report.pdf_font_status()["status"] == "embedded"
    pdf = report.build_pdf_report(sample_report())
    assert b"/FontFile2" in pdf
    assert_pdf_conclusions(pdf, sample_report())


def test_font_initialization_failure_has_actionable_error_and_markdown(monkeypatch):
    from reportlab.pdfbase import pdfmetrics
    monkeypatch.setattr(pdfmetrics, "getRegisteredFontNames", lambda: [])
    monkeypatch.setattr(pdfmetrics, "registerFont", Mock(side_effect=RuntimeError("private detail")))
    status = report.pdf_font_status()
    assert status["status"] == "unavailable"
    final = sample_report()
    with pytest.raises(RuntimeError, match="Markdown 仍可下载"):
        report.build_pdf_report(final)
    assert "静态风险需要复核" in report.build_markdown_report(final)


def test_pdf_failure_preserves_markdown_download_and_ui(tmp_path):
    from streamlit.testing.v1 import AppTest
    source = tmp_path / "model.py"
    source.write_text("value = 1\n", encoding="utf-8")
    state = NS()
    with patch.object(app.st, "session_state", state), patch.object(app, "build_pdf_report", side_effect=RuntimeError("PDF 字体不可用；请恢复字体资源。")):
        app.analyze_project_files([(source.name, source)], generate_ai=False)
    assert state.report_pdf is None and state.report_markdown
    assert "字体" in state.report_error
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30)
    for key, value in vars(state).items():
        at.session_state[key] = value
    at.session_state["active_page"] = "Report"
    at.run()
    assert not at.exception
    assert any("Markdown" in warning.value for warning in at.warning)


@pytest.fixture(scope="module")
def original_state(real_case_files):
    name, files, hashes = real_case_files
    calls = []
    def create(**kwargs):
        raw = kwargs["messages"][1]["content"].split("结构化 Finding 与 analysis_facts（唯一事实依据）：\n")[1]
        facts = json.JSONDecoder().raw_decode(raw)[0]
        calls.append({"characters": sum(len(m["content"]) for m in kwargs["messages"]),
                      "ids": [f["id"] for f in facts["findings"]]})
        payload = {"explanations": [{"finding_id": f["id"], "cause": "请复核此项静态证据在建模路径中的适用条件。",
                   "potential_impact": "实际影响需要结合数据来源与运行对照确认。",
                   "recommendation": "依据原始位置完成复核，保持独立的验证记录。"} for f in facts["findings"]]}
        return NS(choices=[NS(message=NS(content=json.dumps(payload, ensure_ascii=False)))])
    state = NS()
    client = NS(chat=NS(completions=NS(create=create)))
    with patch.object(app.st, "session_state", state), patch.object(agent, "get_client", return_value=client):
        app.analyze_project_files(files, generate_ai=True)
    for alias, path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[alias]
    artifact_dir = os.environ.get("CODEDATA_TEST_ARTIFACT_DIR")
    if artifact_dir:
        out = Path(artifact_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(state.report_markdown, encoding="utf-8")
        if state.report_pdf:
            (out / "report.pdf").write_bytes(state.report_pdf)
        (out / "report.json").write_text(json.dumps(state.report_data, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "inputs-and-prompts.json").write_text(json.dumps({"case": name, "input_sha256": hashes,
            "prompt_batches": calls, "raw_rows": state.data_results[0]["result"]["shape"]["rows"],
            "enhancement_status": state.enhancement_status}, ensure_ascii=False, indent=2), encoding="utf-8")
    return name, state, calls


def test_original_case_static_findings_and_successful_pdf(original_state):
    name, state, calls = original_state
    assert state.report_error is None
    assert state.enhancement_status["ai"]["status"] == "success"
    reader, text = assert_pdf_conclusions(state.report_pdf, state.report_data)
    assert len(reader.pages) >= 2
    assert max(call["characters"] for call in calls) <= 14000
    ids = [fid for call in calls for fid in call["ids"]]
    assert len(ids) == len(set(ids))
    data = state.data_results[0]["result"]
    assert data["shape"]["rows"] == (916 if name == "titanic" else 15411)
    if name == "titanic":
        impacts = [i for i in state.quantified_impacts if i["operation"] == "dropna"]
        assert any((i["before"], i["after"], i["removed"]) == (916, 647, 269) for i in impacts)
        assert any(f["type"] == "Data Leakage Risk" and f["verification_status"] == "static_risk"
                   for f in state.report_data["findings"])
    else:
        duplicates = data["duplicate_split_analysis"]
        assert duplicates["scopes"]["raw_full_row"]["cross_split_duplicate_groups"] == 0
        assert duplicates["modeling_data"]["train_rows"] == 12328
        assert duplicates["modeling_data"]["test_rows"] == 3083
        assert not any(f["rule_id"] == "DQ_CROSS_SPLIT_ENTITY_ID_OVERLAP" for f in state.report_data["findings"])


def test_original_case_streamlit_report_and_ai_pages(original_state):
    from streamlit.testing.v1 import AppTest
    _, state, _ = original_state
    for page in ("AI Review", "Report"):
        at = AppTest.from_file(str(Path(app.__file__)), default_timeout=40)
        for key, value in vars(state).items():
            at.session_state[key] = value
        at.session_state["active_page"] = page
        at.run()
        assert not at.exception
        for line in conclusion_summary_lines(state.report_data):
            assert line in [item.value for item in at.caption]
