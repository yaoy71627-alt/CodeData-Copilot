"""Pinned source snapshots: static project pipeline, no source execution or network."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import app
from consistency_validator import conclusion_summary_lines, finding_summary_line
from report import build_markdown_report


ROOT = Path(__file__).parent / "fixtures/batch_b"


@pytest.fixture(scope="module", params=["ibm", "telco"])
def batch_b_state(request):
    case = request.param
    if case == "ibm":
        files = [(name, ROOT / case / name) for name in ("employee-attrition.ipynb", "emp_attrition.csv")]
    else:
        files = [(p.relative_to(ROOT / case).as_posix(), p) for p in sorted((ROOT / case).rglob("*.py"))]
        files.append(("README.md", ROOT / case / "README.md"))
    before = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files}
    state = SimpleNamespace()
    with patch.object(app.st, "session_state", state):
        app.analyze_project_files(files, generate_ai=False, project_metadata={"project_name": case})
    assert before == {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files}
    assert state.report_error is None and state.report_pdf.startswith(b"%PDF")
    out = os.environ.get("CODEDATA_BATCH_B_ARTIFACT_DIR")
    if out:
        path = Path(out) / case
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.pdf").write_bytes(state.report_pdf)
        (path / "report.md").write_text(state.report_markdown, encoding="utf-8")
        (path / "findings.json").write_text(json.dumps(state.report_data["findings"], ensure_ascii=False, indent=2), encoding="utf-8")
    return case, state


def test_source_snapshots_match_provenance():
    records = json.loads((ROOT / "SOURCES.json").read_text(encoding="utf-8"))["files"]
    assert len(records) == 9
    for item in records:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        assert len(item["commit"]) == 40 and item["commit"] in item["url"]


def test_pinned_case_required_findings_and_semantics(batch_b_state):
    case, state = batch_b_state
    findings = state.report_data["findings"]
    by_rule = {item["rule_id"]: item for item in findings}
    if case == "ibm":
        assert {"CODE_SKLEARN_LEGACY_API", "CODE_SLICE_TEST_SAMPLE"} <= by_rule.keys()
        sample = by_rule["CODE_SLICE_TEST_SAMPLE"]
        assert sample["location"]["cell"] == 109 and sample["location"]["line"] == 2
        assert "1 行" in sample["evidence"] and "部署连通性烟测" in sample["evidence"]
        assert sample["verification_status"] == "needs_review"
        assert state.data_flow["fairness"]["evaluation_exists"] is True
        assert state.data_results[0]["result"]["shape"]["rows"] == 1470
    else:
        required = {"CODE_ONEHOT_SPARSE_API", "CODE_PANDAS_APPEND", "CODE_CHAINED_ASSIGNMENT",
                    "CODE_DUPLICATE_CHECK_CALL", "CODE_UNUSED_STATUS_COMPARISON", "CODE_DOCUMENTED_SPLIT_MISMATCH"}
        assert required <= by_rule.keys()
        assert "CODE_NO_CROSS_VALIDATION" not in by_rule
        assert by_rule["CODE_DUPLICATE_CHECK_CALL"]["location"]["line"] == 69
        assert by_rule["CODE_UNUSED_STATUS_COMPARISON"]["verification_status"] == "confirmed_error"
        assert by_rule["CODE_DOCUMENTED_SPLIT_MISMATCH"]["location"]["line"] == 67
        for identifier in ("CODE_DUPLICATE_CHECK_CALL", "CODE_DOCUMENTED_SPLIT_MISMATCH", "CODE_ONEHOT_SPARSE_API", "CODE_PANDAS_APPEND"):
            assert by_rule[identifier]["verification_status"] == "needs_review"
    # These snapshots have no package pins: never claim runtime incompatibility.
    assert by_rule["CODE_SKLEARN_LEGACY_API" if case == "ibm" else "CODE_ONEHOT_SPARSE_API"]["verification_status"] == "needs_review"


def test_case_canonical_page_markdown_pdf_consistency(batch_b_state):
    _, state = batch_b_state
    final = state.report_data
    assert state.report_markdown == build_markdown_report(final)
    compact = lambda value: "".join(value.split())
    reader = PdfReader(BytesIO(state.report_pdf))
    text = compact("\n".join(page.extract_text() for page in reader.pages))
    for line in conclusion_summary_lines(final):
        assert compact(line) in text
    for finding in final["conclusions"]["finding_rows"]:
        line = finding_summary_line(finding)
        assert line in state.report_markdown
        assert compact(line) in text
    assert len({f["id"] for f in final["findings"]}) == len(final["findings"])
    assert state.patch_text == final["patch_text"]
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30)
    for key, value in vars(state).items():
        at.session_state[key] = value
    at.session_state["active_page"] = "Report"
    at.run()
    assert not at.exception
    assert at.session_state["report_data"]["final_fingerprint"] == final["final_fingerprint"]


def test_dependency_and_document_upload_reaches_final_findings(tmp_path):
    from file_loader import materialize_project_uploads
    class Upload:
        def __init__(self, name, text):
            self.name, self.text = name, text
        def getvalue(self):
            return self.text.encode()
    uploads = [Upload("requirements.txt", "scikit-learn>=1.4"),
               Upload("README.md", "Split: 80% train/20% test."),
               Upload("model.py", "from sklearn.preprocessing import OneHotEncoder\nfrom sklearn.model_selection import train_test_split\nencoder=OneHotEncoder(sparse=True)\na,b=train_test_split(raw,test_size=0.3)")]
    state = SimpleNamespace()
    with materialize_project_uploads(uploads) as files, patch.object(app.st, "session_state", state):
        app.analyze_project_files(files, generate_ai=False)
    items = {f["rule_id"]: f for f in state.report_data["findings"]}
    assert items["CODE_ONEHOT_SPARSE_API"]["verification_status"] == "confirmed_error"
    assert "requirements.txt" in items["CODE_ONEHOT_SPARSE_API"]["evidence"]
    assert "CODE_DOCUMENTED_SPLIT_MISMATCH" in items
    assert not state.patch_text, "Unverified migrations must not become executable patches"
    with pytest.raises(ValueError, match="仅支持"):
        with materialize_project_uploads([Upload("other.txt", "data")]):
            pass
