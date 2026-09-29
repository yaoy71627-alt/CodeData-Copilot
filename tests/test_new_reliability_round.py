"""Regression tests for duplicate artifacts, notebook syntax, split roles and API semantics."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nbformat
import pandas as pd

from code_analyzer import analyze_code_with_ast, analyze_notebook
from consistency_validator import build_analysis_facts, validate_agent_consistency
from data_analyzer import analyze_csv
from data_flow_analyzer import analyze_python_data_flow
from project_dedup import detect_duplicate_artifacts, consolidate_duplicate_code_issues
from repair_validator import validate_repair_suggestions


def _write_notebook(path, source, execution_count=1, output=None):
    notebook = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(source, execution_count=execution_count)
    if output is not None:
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output)]
    notebook.cells = [cell]
    nbformat.write(notebook, path)


class NewReliabilityRoundTests(unittest.TestCase):
    def test_01_exact_duplicate_notebooks_count_issue_once_and_keep_affected_files(self):
        with TemporaryDirectory() as directory:
            first = Path(directory) / "demo.ipynb"
            second = Path(directory) / "demo_copy.ipynb"
            source = "!pip install demo_pkg\nvalue = 1"
            _write_notebook(first, source, execution_count=1, output="first")
            _write_notebook(second, source, execution_count=9, output="second")
            artifacts = detect_duplicate_artifacts([(first.name, first), (second.name, second)])
            raw = []
            for path in (first, second):
                raw.extend({"file": path.name, **entry} for entry in analyze_notebook(path))
            merged = consolidate_duplicate_code_issues(raw, artifacts)
        inline = [
            issue for location in merged for issue in location["issues"]
            if issue["rule_id"] == "NB_INLINE_INSTALL"
        ]
        self.assertEqual(len(inline), 1)
        self.assertEqual(set(inline[0]["affected_files"]), {"demo.ipynb", "demo_copy.ipynb"})
        self.assertEqual(artifacts["groups"][0]["type"], "Duplicate Analysis Artifact")

    def test_02_near_duplicate_notebooks_merge_matching_findings(self):
        common = "!pip install demo_pkg\n" + "\n".join(f"feature_{i} = {i}" for i in range(40))
        changed = common.replace("feature_39 = 39", "feature_39 = 390")
        with TemporaryDirectory() as directory:
            first = Path(directory) / "a.ipynb"
            second = Path(directory) / "b.ipynb"
            _write_notebook(first, common)
            _write_notebook(second, changed)
            artifacts = detect_duplicate_artifacts([(first.name, first), (second.name, second)])
            raw = []
            for path in (first, second):
                raw.extend({"file": path.name, **entry} for entry in analyze_notebook(path))
            merged = consolidate_duplicate_code_issues(raw, artifacts)
        self.assertEqual(artifacts["groups"][0]["type"], "Near-duplicate Notebook")
        inline = [issue for location in merged for issue in location["issues"] if issue["rule_id"] == "NB_INLINE_INSTALL"]
        self.assertEqual(len(inline), 1)
        self.assertEqual(len(inline[0]["affected_files"]), 2)

    def test_03_notebook_magic_and_shell_are_not_python_syntax_errors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "magic.ipynb"
            _write_notebook(path, "%matplotlib inline\n!pip install xxx\nvalue = 1")
            results = analyze_notebook(path)
        rules = {issue["rule_id"] for entry in results for issue in entry["issues"]}
        self.assertNotIn("CODE_SYNTAX_ERROR", rules)
        self.assertIn("NB_IPYTHON_MAGIC", rules)
        self.assertIn("NB_INLINE_INSTALL", rules)

    def test_04_real_python_syntax_error_remains_visible_with_original_line(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.ipynb"
            _write_notebook(path, "%matplotlib inline\nx = (")
            results = analyze_notebook(path)
        issue = next(
            issue for entry in results for issue in entry["issues"]
            if issue["rule_id"] == "CODE_SYNTAX_ERROR"
        )
        self.assertEqual(issue["line"], 2)

    def test_05_train_validation_test_roles_are_detected(self):
        code = """
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)
accuracy_score(y_val, val_pred)
test_pred = model.predict(X_test)
accuracy_score(y_test, test_pred)
"""
        flow = analyze_python_data_flow(code)
        roles = flow["split_role_analysis"]
        self.assertTrue(roles["train_detected"])
        self.assertTrue(roles["validation_detected"])
        self.assertTrue(roles["test_detected"])
        rules = {issue["rule_id"] for issue in analyze_code_with_ast(code)}
        self.assertNotIn("CODE_NO_VALIDATION_SPLIT", rules)
        self.assertNotIn("CODE_NO_EVALUATION", rules)

    def test_06_repeated_test_comparison_is_model_selection_leakage(self):
        code = """
for n in [10, 50, 100]:
    model = RandomForestClassifier(n_estimators=n, random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
"""
        issues = analyze_code_with_ast(code)
        issue = next(item for item in issues if item["rule_id"] == "CODE_TEST_SET_OVERUSE")
        self.assertEqual(issue["type"], "Test-set Overuse / Model Selection Leakage")

    def test_07_discrete_count_feature_downgrades_iqr_to_distribution_review(self):
        values = [1, 2] * 20 + [0, 3, 5, 6]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            pd.DataFrame({"TrainingTimesLastYear": values}).to_csv(path, index=False)
            result = analyze_csv(path)
        rules = {issue["rule_id"] for issue in result["issues"]}
        self.assertIn("DQ_DISCRETE_DISTRIBUTION_REVIEW", rules)
        self.assertNotIn("DQ_POTENTIAL_EXTREME_VALUES", rules)
        issue = next(item for item in result["issues"] if item["rule_id"] == "DQ_DISCRETE_DISTRIBUTION_REVIEW")
        self.assertEqual(issue["action"], "Review")

    def test_08_continuous_salary_still_uses_outlier_detection(self):
        values = list(range(50_000, 50_030)) + [5_000_000]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "salary.csv"
            pd.DataFrame({"salary": values}).to_csv(path, index=False)
            result = analyze_csv(path)
        self.assertIn("DQ_POTENTIAL_EXTREME_VALUES", {issue["rule_id"] for issue in result["issues"]})
        self.assertEqual(result["numeric_variable_types"]["salary"]["type"], "continuous numeric")

    def test_09_custom_transformer_refit_during_transform_is_detected_once(self):
        code = """
class PipelineLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return LabelEncoder().fit_transform(X)
"""
        issues = analyze_code_with_ast(code)
        refits = [item for item in issues if item["rule_id"] == "CODE_TRANSFORMER_REFIT_DURING_TRANSFORM"]
        self.assertEqual(len(refits), 1)
        self.assertEqual(refits[0]["impacts"], ["inconsistent category mapping", "validation/test instability"])
        self.assertNotIn("CODE_PREPROCESS_FULL_DATA", {item["rule_id"] for item in issues})

    def test_10_correct_custom_transformer_is_not_flagged(self):
        code = """
class PipelineLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoder_ = LabelEncoder().fit(X)
        return self
    def transform(self, X):
        return self.encoder_.transform(X)
"""
        self.assertNotIn(
            "CODE_TRANSFORMER_REFIT_DURING_TRANSFORM",
            {item["rule_id"] for item in analyze_code_with_ast(code)},
        )

    def test_11_wrong_aif360_metric_method_is_rejected(self):
        flow = analyze_python_data_flow("metric = BinaryLabelDatasetMetric(dataset)")
        suggestions = [{
            "problem": "Fairness repair",
            "after_code": "metric.equal_opportunity_difference()",
            "corrected_code": "metric.equal_opportunity_difference()",
        }]
        checked = validate_repair_suggestions(suggestions, flow, [])
        self.assertEqual(checked[0]["validation_status"], "Failed Validation")
        self.assertIsNone(checked[0]["after_code"])

    def test_12_correct_classification_metric_method_and_inputs_pass(self):
        flow = analyze_python_data_flow(
            "metric = ClassificationMetric(dataset_true, dataset_pred, "
            "unprivileged_groups=groups_a, privileged_groups=groups_b)"
        )
        suggestions = [{
            "problem": "Fairness repair",
            "after_code": "metric.equal_opportunity_difference()",
            "corrected_code": "metric.equal_opportunity_difference()",
        }]
        checked = validate_repair_suggestions(suggestions, flow, [])
        self.assertEqual(checked[0]["validation_status"], "Fully Validated")

    def test_13_fairness_before_reweighing_after_is_recognized(self):
        code = """
metric_before = BinaryLabelDatasetMetric(dataset_before)
before = metric_before.mean_difference()
reweigher = Reweighing(unprivileged_groups=groups_a, privileged_groups=groups_b)
dataset_after = reweigher.fit_transform(dataset_before)
metric_after = BinaryLabelDatasetMetric(dataset_after)
after = metric_after.mean_difference()
"""
        flow = analyze_python_data_flow(code)
        fairness = flow["fairness"]
        self.assertTrue(fairness["evaluation_exists"])
        self.assertTrue(fairness["before_after_comparison"])
        self.assertEqual(fairness["coverage"], "Fairness evaluation exists, but metric coverage is limited.")
        facts = build_analysis_facts([], [], flow)
        corrected, _ = validate_agent_consistency("No fairness evaluation was performed.", facts)
        self.assertNotIn("No fairness evaluation", corrected)


if __name__ == "__main__":
    unittest.main()
