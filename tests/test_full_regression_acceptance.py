"""Full Notebook -> analyzer -> mocked Qwen -> final UI/MD/PDF acceptance.

Fixtures below are constructed regression cases, NOT the missing original
Titanic/Cardekho datasets. Notebook cells are parsed, never executed.
"""
import copy
import hashlib
import html
import json
import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nbformat
import pandas as pd
import pytest
from pypdf import PdfReader

import agent
import app
from consistency_validator import (
    conclusion_summary_lines, finding_summary_line, is_final_report,
    validate_report_consistency,
)
from patch_generator import build_unified_patch
from repair_validator import validate_repair_suggestions
from report import build_markdown_report, build_pdf_report

ROOT = Path(__file__).resolve().parents[1]
TITANIC_CELLS = [
    '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("titanic_constructed.csv")''',
    '''feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
model_data = df[feature_cols + ["Survived"]].copy()
model_data = model_data.dropna()
X = model_data[feature_cols]
y = model_data["Survived"]
scaler = StandardScaler()
X = scaler.fit_transform(X)''',
    '''X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)''',
]
CARDEKHO_CELLS = [
    '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
df = pd.read_csv("cardekho_constructed.csv")''',
    '''feature_cols = ["year", "km_driven"]
X = df[feature_cols]
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = mean_absolute_error(y_test, y_pred)''',
]


def write_case(directory, name, cells, frame):
    directory.mkdir()
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("构造回归案例；非原始案例数据。"),
        *[nbformat.v4.new_code_cell(source) for source in cells],
    ])
    nb = directory / f"{name}.ipynb"
    csv = directory / f"{name}_constructed.csv"
    nbformat.write(notebook, nb)
    frame.to_csv(csv, index=False)
    return [(nb.name, nb), (csv.name, csv)]


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("notebook_acceptance")
    n = 80
    titanic = pd.DataFrame({
        "PassengerId": range(1, n + 1), "Survived": [i % 2 for i in range(n)],
        "Pclass": [i % 3 + 1 for i in range(n)],
        "Age": [None if i >= 64 else 18 + i / 2 for i in range(n)],
        "SibSp": [i % 3 for i in range(n)], "Parch": [i % 2 for i in range(n)],
        "Fare": [10 + i / 10 for i in range(n)], "Cabin": [None] * n,
    })
    titanic.loc[0, "Fare"] = -10
    titanic.loc[1, "Age"] = -5
    cardekho = pd.DataFrame({
        "row_number": range(n), "car_name": ["sedan", "suv"] * (n // 2),
        "year": [2015 + i % 4 for i in range(n)],
        "km_driven": [20_000 + (i % 16) * 1000.25 for i in range(n)],
        "selling_price": [200_000 + i * 1007 for i in range(n)],
    })
    # An unusual but positive mileage is a statistical candidate, not a business error.
    cardekho.loc[n - 1, "km_driven"] = 1_000_000
    return {
        "titanic": write_case(root / "titanic", "titanic", TITANIC_CELLS, titanic),
        "cardekho": write_case(root / "cardekho", "cardekho", CARDEKHO_CELLS, cardekho),
    }


def prompt_facts(kwargs):
    prompt = kwargs["messages"][1]["content"]
    raw = prompt.split("结构化 Finding 与 analysis_facts（唯一事实依据）：\n", 1)[1]
    return json.JSONDecoder().raw_decode(raw)[0]


def complete_explanation(finding):
    return {"finding_id": finding["id"],
            "cause": "请核查该证据对应的实际执行路径。",
            "potential_impact": "实际影响需要结合运行对照确认。",
            "recommendation": "根据原始位置复核触发条件并保留验证记录。"}


def mocked_client(payload=None, exception=None):
    def completion(**kwargs):
        if exception is not None:
            raise exception
        facts = prompt_facts(kwargs)
        data = payload(facts) if payload else {"explanations": [
            complete_explanation(f) for f in facts["findings"]]}
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    create = Mock(side_effect=completion)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


def assert_bounded_without_retry(create):
    from qwen_runtime import load_settings
    settings = load_settings()
    assert 1 <= create.call_count <= settings.max_batches * (settings.retry_count + 1)
    for call in create.call_args_list:
        ids = [f["id"] for f in prompt_facts(call.kwargs)["findings"]]
        assert len(ids) == len(set(ids)), "A retry batch must not contain duplicate Finding IDs"


def digest_inputs(files):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files}


def analyze(files, client=None, ai=True):
    before = digest_inputs(files)
    state = SimpleNamespace()
    with patch.object(app.st, "session_state", state):
        if client is None:
            app.analyze_project_files(files, generate_ai=ai)
        else:
            with patch.object(agent, "get_client", return_value=client):
                app.analyze_project_files(files, generate_ai=ai)
    assert digest_inputs(files) == before, "Analysis must not modify uploaded Notebook/CSV"
    assert state.report_error is None
    assert is_final_report(state.report_data)
    assert state.report_pdf.startswith(b"%PDF") and b"%%EOF" in state.report_pdf[-30:]
    assert state.report_markdown
    return state


@pytest.fixture(scope="module")
def states(cases):
    result = {}
    for name, files in cases.items():
        client, create = mocked_client()
        result[name] = analyze(files, client)
        assert_bounded_without_retry(create)
    return result


def pdf_text(report):
    from reportlab.platypus import Paragraph
    captured = []
    def capture(text, *args, **kwargs):
        captured.append(html.unescape(re.sub(r"<[^>]+>", "", text)))
        return Paragraph(text, *args, **kwargs)
    with patch("reportlab.platypus.Paragraph", side_effect=capture):
        pdf = build_pdf_report(report)
    assert pdf.startswith(b"%PDF") and b"/FontFile2" in pdf
    return captured


@pytest.mark.parametrize("case", ["titanic", "cardekho"])
def test_real_notebook_ingestion_with_successful_mock_qwen(states, case):
    state = states[case]
    assert state.enhancement_status["knowledge"]["status"] == "success"
    assert state.enhancement_status["knowledge"]["sources"]
    assert state.enhancement_status["knowledge"]["rules"]
    assert {k: state.enhancement_status["ai"][k] for k in ("status", "reason", "used_knowledge")} == {
        "status": "success", "reason": "Qwen Review 已生成。", "used_knowledge": True}
    coverage = state.report_data["review_coverage"]
    assert coverage["total"] > 0
    assert coverage["ai_explained"] == coverage["total"]
    assert coverage["deterministic_filled"] == 0
    assert state.analysis_facts["detected_split"] and state.analysis_facts["evaluation_detected"]
    assert state.analysis_facts["datasets"][0]["rows"] == 80


def test_titanic_findings_and_repairs_match_actual_notebook(states, cases):
    state = states["titanic"]
    findings = state.report_data["findings"]
    leak = next(f for f in findings if f["type"] == "Data Leakage Risk")
    # Markdown occupies Cell 1; the fit_transform is Cell 3, Line 7.
    assert leak["location"]["file"] == "titanic.ipynb"
    assert (leak["location"]["cell"], leak["location"]["line"]) == (3, 7)
    assert leak["verification_status"] == "static_risk"
    assert any(f["origin"] == "data" and f["location"]["column"] == "Age" for f in findings)
    assert any(f["origin"] == "data" and f["location"]["column"] == "Fare" for f in findings)
    assert state.analysis_facts["model_features"] == ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    assert "Cabin" not in state.analysis_facts["model_features"]
    relevant = [r for r in state.repair_suggestions if r["problem"] in {
        "Data Leakage Risk", "Potential Data Leakage", "Missing Value Handling Risk"}]
    assert relevant
    for repair in relevant:
        assert repair["validation_status"] == "Recommendation Only"
        assert not repair.get("after_code") and not repair.get("executable")
        assert "Cabin" not in repair.get("referenced_columns", [])
    assert not state.patch_text
    assert digest_inputs(cases["titanic"])


def test_cardekho_reused_features_and_outlier_are_not_confirmed_business_errors(states):
    state = states["cardekho"]
    result = state.data_results[0]["result"]
    assert result["duplicate_rows"] == 0
    duplicate = result["duplicate_split_analysis"]
    assert duplicate["cross_split_duplicate_groups"] > 0
    assert duplicate["duplicate_kind"] == "model_input_features"
    assert "business_id" not in duplicate["scopes"]
    findings = state.report_data["findings"]
    reused = next(f for f in findings if f["rule_id"] == "DQ_CROSS_SPLIT_FEATURE_COMBINATIONS")
    assert (reused["severity"], reused["confidence"], reused["verification_status"]) == ("Low", "Medium", "needs_review")
    outliers = [f for f in findings if f["rule_id"] == "DQ_POTENTIAL_EXTREME_VALUES" and f["location"]["column"] == "km_driven"]
    assert outliers, "The fixture must actually exercise statistical anomaly handling"
    assert all(f["issue_nature"] == "Statistical Anomaly" and f["detected_count"] == 1 for f in outliers)
    assert all(f["verification_status"] == "needs_review" for f in outliers)
    assert not any(f["rule_id"] == "DQ_CROSS_SPLIT_DUPLICATES" for f in findings)


@pytest.mark.parametrize("case", ["titanic", "cardekho"])
def test_adversarial_qwen_cannot_add_facts_or_omit_findings(cases, states, case):
    def malicious(facts):
        first = facts["findings"][0]
        return {"explanations": [{**complete_explanation(first),
            "cause": "请核查该证据对应的实际执行路径。训练集有9999行。存在医疗风险。",
            "potential_impact": "模型准确率显著下降。性能严重高估。相同车型说明已确认实体泄漏。",
            "recommendation": "严重程度应升级为Critical。建议使用 ghost_scaler.fit(X)。",
            "severity": "Critical", "count": 9999, "verification_status": "confirmed_error",
            "location": {"file": "invented.py", "line": 9999}},
            {"finding_id": "F-forged", "cause": "新增确定性错误"}],
            "unverified_suggestions": ["建议核查部署环境中的依赖一致性。"]}
    client, create = mocked_client(malicious)
    state = analyze(cases[case], client)
    assert_bounded_without_retry(create)
    assert state.report_data["findings"] == states[case].report_data["findings"]
    coverage = state.report_data["review_coverage"]
    assert coverage["ai_explained"] == 0
    assert coverage["deterministic_filled"] == coverage["total"]
    assert coverage["priority_filled"] == coverage["priority_total"]
    text = state.report_markdown + "\n".join(pdf_text(state.report_data))
    for forbidden in ("9999", "医疗风险", "准确率显著下降", "性能严重高估", "已确认实体泄漏",
                      "ghost_scaler", "invented.py", "F-forged", "新增确定性错误",
                      "Unable to verify automatically"):
        assert forbidden not in text
    assert "请核查该证据对应的实际执行路径。" not in state.agent_report
    assert "待验证建议（不计入原始检测数量）" in state.agent_report
    for finding in state.report_data["findings"]:
        assert finding["id"] in state.agent_report
    for item in coverage["items"]:
        assert item["id"] in state.agent_report
        if item["priority"]:
            for section in ("## 3. 问题原因", "## 4. 影响分析", "## 5. 优化建议"):
                body = state.agent_report.split(section, 1)[1].split("\n\n## ", 1)[0]
                assert item["id"] in body


@pytest.mark.parametrize("payload", [
    lambda facts: {"explanations": []},
    lambda facts: {"explanations": [complete_explanation(facts["findings"][0])]},
    lambda facts: "{not valid JSON",
])
def test_empty_partial_or_malformed_qwen_is_completed_without_extra_calls(cases, payload):
    client, create = mocked_client(payload)
    state = analyze(cases["titanic"], client)
    assert_bounded_without_retry(create)
    coverage = state.report_data["review_coverage"]
    assert coverage["total"] == coverage["ai_explained"] + coverage["deterministic_filled"]
    assert coverage["deterministic_filled"] > 0
    for finding in state.report_data["findings"]:
        assert state.report_data["display_ids"]["findings"][finding["id"]] in state.report_markdown
    assert "结构化事实补齐" in state.agent_report


@pytest.mark.parametrize("fault,knowledge,ai_status", [
    ("missing_directory", "read_failed", "success"),
    ("unreadable_file", "read_failed", "success"),
    ("retrieval", "retrieval_failed", "success"),
    ("missing_key", "success", "missing_api_key"),
    ("timeout", "success", "timeout"),
    ("api_exception", "success", "api_failed"),
    ("both_unavailable", "retrieval_failed", "timeout"),
])
def test_fault_injection_keeps_notebook_static_findings_and_both_exports(
        cases, states, tmp_path, monkeypatch, fault, knowledge, ai_status):
    if fault == "missing_directory":
        monkeypatch.setattr(agent, "KNOWLEDGE_DIR", tmp_path / "absent")
    elif fault == "unreadable_file":
        read_text = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path.parent == agent.KNOWLEDGE_DIR:
                raise PermissionError("private diagnostic payload")
            return read_text(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", unreadable)
    elif fault in {"retrieval", "both_unavailable"}:
        monkeypatch.setattr(agent, "retrieve_relevant_knowledge", Mock(side_effect=RuntimeError("private diagnostic payload")))
    if fault == "missing_key":
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        # A missing key must fail before constructing an SDK client.
        with patch("openai.OpenAI") as ctor:
            state = analyze(cases["titanic"])
        ctor.assert_not_called()
    else:
        exception = None
        if fault in {"timeout", "both_unavailable"}:
            from openai import APITimeoutError
            exception = APITimeoutError(request=SimpleNamespace(
                method="POST", url="https://example.invalid"))
        elif fault == "api_exception":
            exception = RuntimeError("private diagnostic payload")
        client, create = mocked_client(exception=exception)
        state = analyze(cases["titanic"], client)
        assert_bounded_without_retry(create)
    baseline = states["titanic"]
    for key in ("code_result", "data_results", "analysis_facts", "health"):
        assert getattr(state, key) == getattr(baseline, key)
    assert state.report_data["findings"] == baseline.report_data["findings"]
    assert state.enhancement_status["knowledge"]["status"] == knowledge
    assert state.enhancement_status["ai"]["status"] == ai_status
    assert "private diagnostic payload" not in state.report_markdown
    assert knowledge in state.report_markdown and ai_status in state.report_markdown
    if ai_status != "success":
        assert state.report_data["review_coverage"]["ai_explained"] == 0
        assert "完整静态诊断" in state.report_markdown


@pytest.mark.parametrize("after,expected", [
    ("score = missing_predictions", "Failed Validation"),
    ('score = X_train["Cabin"]', "Failed Validation"),
    ("score = model.not_a_real_api(X_test)", "Failed Validation"),
])
def test_invalid_notebook_repair_is_rejected_in_final_report(states, after, expected):
    state = states["titanic"]
    suggestion = {"file": "titanic.ipynb", "cell": 4, "line": 6,
                  "before_code": "score = accuracy_score(y_test, y_pred)",
                  "after_code": after}
    checked = validate_repair_suggestions([suggestion], state.data_flow, state.data_results)
    assert checked[0]["validation_status"] == expected
    assert not checked[0]["executable"] and not checked[0]["after_code"]
    assert not build_unified_patch(checked)
    report = copy.deepcopy(state.report_data)
    report["repair_suggestions"] = checked
    report["patch_text"] = "forged stale patch"
    final = validate_report_consistency(report)
    assert not final["patch_text"] and not final["view"]["patch_text"]
    assert not final["conclusions"]["patch_available"]
    assert "forged stale patch" not in build_markdown_report(final)
    assert "forged stale patch" not in "\n".join(pdf_text(final))


@pytest.mark.parametrize("case", ["titanic", "cardekho"])
def test_notebook_ui_markdown_pdf_share_counts_metadata_and_repair_status(states, case):
    from streamlit.testing.v1 import AppTest
    state = states[case]
    final = state.report_data
    md, paragraphs = state.report_markdown, pdf_text(final)
    counts = final["conclusions"]["counts"]
    assert counts["total"] == counts["confirmed_error"] + counts["static_risk"] + counts["needs_review"]
    assert counts["total"] == final["review_coverage"]["total"]
    for line in conclusion_summary_lines(final):
        assert line in md and line in paragraphs
    for row in final["conclusions"]["finding_rows"]:
        assert finding_summary_line(row) in md
        assert finding_summary_line(row) in paragraphs
    for page in ("Code Analysis", "Data Quality", "AI Review", "Repair", "Report"):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        for key, value in vars(state).items():
            at.session_state[key] = copy.deepcopy(value)
        at.session_state["active_page"] = page
        at.session_state["patch_text"] = "forged stale patch"
        at.run()
        assert not at.exception
        assert at.session_state["report_data"]["findings"] == final["findings"]
        for line in conclusion_summary_lines(final):
            assert line in [c.value for c in at.caption]
        if page == "AI Review":
            assert next(m for m in at.metric if m.label == "Detected problems").value == str(counts["total"])
        assert "forged stale patch" not in "\n".join(str(m.value) for m in at.markdown)


@pytest.mark.parametrize("case", ["titanic", "cardekho"])
def test_actual_pdf_bytes_retain_chinese_conclusions_and_notebook_locations(states, case):
    state = states[case]
    reader = PdfReader(BytesIO(state.report_pdf))
    extracted = "\n".join(page.extract_text() for page in reader.pages)
    compact = lambda value: re.sub(r"\s+", "", value)
    text = compact(extracted)
    assert len(reader.pages) > 0
    for line in conclusion_summary_lines(state.report_data):
        assert compact(line) in text
    for row in state.report_data["conclusions"]["finding_rows"]:
        assert compact(finding_summary_line(row)) in text


@pytest.mark.parametrize("receiver", ["X_train", "X_test"])
def test_python_split_subset_never_borrows_unused_csv_field_for_patch(receiver):
    from data_flow_analyzer import analyze_python_data_flow
    source = '''import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("train.csv")
X = df[["Age"]]
X_train, X_test = train_test_split(X, random_state=42)
value = 1
'''
    flow = analyze_python_data_flow(source, "model.py")
    item = {"file": "model.py", "line": 6, "before_code": "value = 1",
            "after_code": f'value = {receiver}["Cabin"]'}
    data = [{"file": "train.csv", "result": {"data_types": {"Age": "float64", "Cabin": "object"}}}]
    result = validate_repair_suggestions([item], flow, data)
    assert result[0]["validation_status"] == "Failed Validation"
    assert "Cabin" in result[0]["validation_detail"]
    assert not result[0]["after_code"] and not result[0]["executable"]
    assert not build_unified_patch(result)


def test_unresolved_receiver_schema_cannot_borrow_dataset_columns():
    from data_flow_analyzer import analyze_python_data_flow
    source = '''import pandas as pd
df = pd.read_csv("train.csv")
unresolved = object()
value = 1
'''
    flow = analyze_python_data_flow(source, "model.py")
    item = {"file": "model.py", "line": 4, "before_code": "value = 1",
            "after_code": 'value = unresolved["Age"]'}
    data = [{"file": "train.csv", "result": {"data_types": {"Age": "float64"}}}]
    result = validate_repair_suggestions([item], flow, data)
    assert result[0]["validation_status"] == "Syntax Valid Only"
    assert not result[0]["executable"] and not build_unified_patch(result)
