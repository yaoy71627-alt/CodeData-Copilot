"""Targeted regressions for the Titanic/CarDekho/IBM reliability fixes.

Fixtures are synthetic on purpose: rules must follow lineage and field scope,
not project names or memorized case counts.
"""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import ast
import zipfile

import nbformat
import pandas as pd
from sklearn.model_selection import train_test_split

from code_analyzer import analyze_code_with_ast, analyze_notebook
from data_analyzer import analyze_csv
from data_flow_analyzer import (
    analyze_notebook_data_flow,
    analyze_python_data_flow,
    merge_data_flows,
    reconcile_code_issues_with_flow,
)
from github_loader import _scan_repository
from notebook_syntax import preprocess_notebook_source
from project_dedup import consolidate_duplicate_code_issues
from repair_validator import validate_repair_suggestions


def _write_notebook(path, sources):
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(source, execution_count=index)
        for index, source in enumerate(sources, 1)
    ]
    nbformat.write(notebook, path)


def test_percent_formatting_continuation_is_python_but_real_magic_remains():
    source = 'message = (\n    "employee: %s"\n    % employee_name\n)\n%matplotlib inline\nvalue = 1'
    processed = preprocess_notebook_source(source)

    ast.parse(processed["code"])
    assert processed["code"].splitlines()[2].lstrip().startswith("% employee_name")
    assert [(item["subtype"], item["line"]) for item in processed["constructs"]] == [
        ("line_magic", 5)
    ]


def test_cleanup_is_replayed_before_split_duplicate_scopes_are_calculated():
    frame = pd.DataFrame({
        "Ticket": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        "feature": [1, 1, 2, None, 3, 3, 4, None, 5, 5],
        "target": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    })
    code = """
features_a = ["feature"]
features_b = []
features = features_a + features_b
model_data = df[features + ["target"]]
model_data = model_data.dropna()
X = model_data[features]
y = model_data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=11
)
"""
    flow = analyze_python_data_flow(code)
    with TemporaryDirectory() as directory:
        csv_path = Path(directory) / "records.csv"
        frame.to_csv(csv_path, index=False)
        result = analyze_csv(csv_path, data_flow=flow)

    cleaned = frame.dropna(subset=["feature"])
    train_index, test_index = train_test_split(
        list(cleaned.index), test_size=0.5, random_state=11
    )
    train_ids = set(frame.loc[train_index, "Ticket"])
    test_ids = set(frame.loc[test_index, "Ticket"])
    expected_business_overlap = len(train_ids & test_ids)

    duplicate = result["duplicate_split_analysis"]
    assert duplicate["status"] == "Exact"
    assert duplicate["modeling_data"]["raw_rows"] == 10
    assert duplicate["modeling_data"]["cleaned_rows"] == 8
    assert duplicate["scope"] == "business_id"
    assert duplicate["fields"] == ["Ticket"]
    assert duplicate["cross_split_duplicate_groups"] == expected_business_overlap
    assert duplicate["scopes"]["model_input_row"]["fields"] == ["feature"]


def test_chained_selection_cleanup_and_iloc_encoding_keep_row_lineage():
    frame = pd.DataFrame({
        "record_name": ["a", "a", "b", "b", "c", "c"],
        "category": ["x", "x", "y", "y", "z", "z"],
        "value": [1, 1, 2, None, 3, 3],
        "target": [0, 0, 1, 1, 0, 0],
    })
    code = """
model_data = df[["category", "value", "target"]].dropna()
encoded = pd.get_dummies(model_data, drop_first=True)
X = encoded.iloc[:, 1:]
y = encoded.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=7)
"""
    flow = analyze_python_data_flow(code)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "encoded.csv"
        frame.to_csv(path, index=False)
        result = analyze_csv(path, data_flow=flow)

    duplicate = result["duplicate_split_analysis"]
    assert duplicate["status"] == "Exact"
    assert duplicate["modeling_data"]["cleaned_rows"] == 5
    assert duplicate["modeling_data"]["model_input_variable"] == "X"
    assert duplicate["scope"] == "business_id"
    assert duplicate["fields"] == ["record_name"]


def test_literal_slices_and_cross_cell_test_evaluation_are_project_facts():
    sources = [
        "cut = 6\nX_train = X[:cut]\nX_test = X[cut:]\ny_train = y[:cut]\ny_test = y[cut:]",
        "model = RandomForestClassifier(random_state=1)\nmodel.fit(X_train, y_train)",
        "prediction = model.predict(X_test)\naccuracy_score(y_test, prediction)",
    ]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sliced.ipynb"
        _write_notebook(path, sources)
        flow = analyze_notebook_data_flow(path)
        findings = analyze_notebook(path)

    assert flow["split"]["method"] == "slice"
    assert flow["split_role_analysis"]["train_detected"]
    assert flow["split_role_analysis"]["test_detected"]
    assert flow["evaluation_detected"]
    rules = {issue["rule_id"] for item in findings for issue in item["issues"]}
    assert "CODE_NO_VALIDATION_SPLIT" not in rules
    assert "CODE_NO_EVALUATION" not in rules
    assert "CODE_TRAINING_ONLY_EVALUATION" not in rules


def test_cross_file_fit_and_test_evaluation_are_merged_before_absence_rules():
    training = """
X_train = X[:8]
X_test = X[8:]
y_train = y[:8]
y_test = y[8:]
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
"""
    evaluation = "prediction = model.predict(X_test)\nf1_score(y_test, prediction)"
    raw = [
        {"file": "train.py", "issues": analyze_code_with_ast(training)},
        {"file": "evaluate.py", "issues": analyze_code_with_ast(evaluation)},
    ]
    project_flow = merge_data_flows([
        analyze_python_data_flow(training, "train.py"),
        analyze_python_data_flow(evaluation, "evaluate.py"),
    ])
    reconciled = reconcile_code_issues_with_flow(raw, project_flow)
    rules = {issue["rule_id"] for item in reconciled for issue in item["issues"]}

    assert project_flow["split"]["detected"]
    assert project_flow["evaluation_detected"]
    assert project_flow["metric_coverage"]["detected"] == ["f1"]
    assert "CODE_NO_VALIDATION_SPLIT" not in rules
    assert "CODE_NO_EVALUATION" not in rules


def test_excluded_export_index_uses_real_uniqueness_and_is_informational():
    frame = pd.DataFrame({"Unnamed: 0": [10, 11, 12, 13], "feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    flow = analyze_python_data_flow('X = df[["feature"]]\ny = df["target"]')
    with TemporaryDirectory() as directory:
        path = Path(directory) / "data.csv"
        frame.to_csv(path, index=False)
        result = analyze_csv(path, data_flow=flow)

    issue = next(item for item in result["issues"] if item["rule_id"] == "DQ_POTENTIAL_INDEX_COLUMN")
    assert issue["count"] == 4
    assert issue["rate"] == 1.0
    assert issue["model_input_status"] == "Excluded"
    assert issue["action"] == "Informational"
    assert issue["score_impact"] == 0


def test_explicit_ranges_categories_and_nonnegative_constraints_are_traceable():
    frame = pd.DataFrame({
        "age": [20, 150],
        "price": [10, -1],
        "segment": ["retail", "invalid"],
    })
    constraints = {
        "source": "contract-v3",
        "columns": {
            "age": {"min": 0, "max": 120},
            "price": {"non_negative": True},
            "segment": {"allowed_values": ["retail", "business"]},
        },
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rules.csv"
        frame.to_csv(path, index=False)
        result = analyze_csv(path, constraints=constraints)

    confirmed = [item for item in result["issues"] if item["rule_id"] == "DQ_CONSTRAINT_VIOLATION"]
    assert {item["column"] for item in confirmed} == {"age", "price", "segment"}
    assert all(item["confidence"] == "High" for item in confirmed)
    assert all("contract-v3" in item["issue_origin"] for item in confirmed)


def test_repair_field_is_parsed_and_rejected_when_receiver_does_not_have_it():
    flow = analyze_python_data_flow('model_data = df[["Age"]]\nmodel_data = model_data.dropna()')
    suggestion = {
        "problem": "Missing Value Handling Risk",
        "after_code": "model_data.fillna({'Cabin': model_data['Cabin'].mode().iloc[0]})",
        "corrected_code": "model_data.fillna({'Cabin': model_data['Cabin'].mode().iloc[0]})",
        "scope": "model_input",
        "line": 2,
    }
    checked = validate_repair_suggestions(
        [suggestion], flow,
        [{"result": {"data_types": {"Age": "float64", "Cabin": "object"}}}],
    )[0]

    assert checked["validation_status"] == "Failed Validation"
    assert checked["after_code"] is None
    assert "Cabin" in checked["validation_detail"]


def test_near_duplicate_root_cause_merges_despite_literal_change():
    artifacts = {
        "groups": [{
            "type": "Near-duplicate Notebook", "primary": "a.ipynb",
            "files": ["a.ipynb", "b.ipynb"], "similarity": 0.97,
        }]
    }
    raw = [
        {"file": "a.ipynb", "cell": 3, "issues": [{
            "rule_id": "CODE_TEST_SET_OVERUSE", "type": "Test-set Overuse / Model Selection Leakage",
            "root_cause_id": "test-selection:41", "source_snippet": "score_a = metric(y_test, p1)",
        }]},
        {"file": "b.ipynb", "cell": 3, "issues": [{
            "rule_id": "CODE_TEST_SET_OVERUSE", "type": "Test-set Overuse / Model Selection Leakage",
            "root_cause_id": "test-selection:44", "source_snippet": "result = custom_metric(labels_test, prediction_b)",
        }]},
    ]
    merged = consolidate_duplicate_code_issues(raw, artifacts)
    issues = [issue for item in merged for issue in item["issues"]]

    assert len(issues) == 1
    assert set(issues[0]["affected_files"]) == {"a.ipynb", "b.ipynb"}


def test_nested_zip_source_is_safely_loaded_and_custom_transformer_is_checked():
    source = b"""
class PipelineLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return LabelEncoder().fit_transform(X)
"""
    with TemporaryDirectory() as directory:
        root = Path(directory) / "repository"
        root.mkdir()
        inner = BytesIO()
        with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("src/encoder.py", source)
        with zipfile.ZipFile(root / "dependency.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("vendor/source.zip", inner.getvalue())
        files = _scan_repository(root)
        nested = next((name, path) for name, path in files if name.endswith("encoder.py"))
        issues = analyze_code_with_ast(nested[1].read_text(encoding="utf-8"))

    assert "dependency.zip!/vendor/source.zip!/src/encoder.py" == nested[0]
    assert [item for item in issues if item["rule_id"] == "CODE_TRANSFORMER_REFIT_DURING_TRANSFORM"]
