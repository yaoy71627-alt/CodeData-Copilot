"""Regression tests for the eight report-reliability gaps."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nbformat
import pandas as pd

import agent
from code_analyzer import analyze_code_with_ast, analyze_notebook
from consistency_validator import (
    build_analysis_facts,
    validate_agent_consistency,
    validate_report_consistency,
)
from data_analyzer import analyze_csv
from data_flow_analyzer import (
    analyze_notebook_data_flow,
    analyze_python_data_flow,
    enrich_model_features,
    quantify_data_impacts,
    reconcile_code_issues_with_flow,
)
from repair_validator import validate_repair_suggestions


CLASSIFICATION_CODE = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("train.csv")
features = ["age", "fare"]
X = df[features]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline = Pipeline([])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)
"""


class ReliabilityGapTests(unittest.TestCase):
    def test_01_complete_classification_flow_has_split_test_and_evaluation(self):
        flow = analyze_python_data_flow(CLASSIFICATION_CODE, "model.py")
        self.assertTrue(flow["split"]["detected"])
        self.assertIn("X_test", flow["split"]["test"])
        self.assertEqual(flow["fit"][-1]["X"], "X_train")
        self.assertEqual(flow["predict"][-1]["X"], "X_test")
        self.assertEqual(flow["task_type"]["type"], "Classification")
        self.assertEqual(flow["metric_coverage"]["detected"], ["accuracy"])
        self.assertEqual(flow["metric_coverage"]["status"], "Metric Coverage Limited")
        reconciled = reconcile_code_issues_with_flow(
            [{"file": "model.py", "issues": analyze_code_with_ast(CLASSIFICATION_CODE)}],
            flow,
        )
        rules = {issue["rule_id"] for item in reconciled for issue in item["issues"]}
        self.assertNotIn("CODE_NO_VALIDATION_SPLIT", rules)
        self.assertNotIn("CODE_NO_EVALUATION", rules)

    def test_02_complete_regression_flow_recognizes_mae(self):
        code = """
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
model = RandomForestRegressor(random_state=7)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
"""
        flow = analyze_python_data_flow(code)
        self.assertEqual(flow["task_type"]["type"], "Regression")
        self.assertEqual(flow["metric_coverage"]["detected"], ["mae"])
        self.assertEqual(flow["metric_coverage"]["status"], "Metric Coverage Limited")
        rules = {item["rule_id"] for item in analyze_code_with_ast(code)}
        self.assertNotIn("CODE_NO_EVALUATION", rules)
        self.assertIn("CODE_SINGLE_REGRESSION_METRIC", rules)

    def test_03_notebook_flow_crosses_cells(self):
        notebook = nbformat.v4.new_notebook()
        sources = [
            'df = pd.read_csv("train.csv")',
            'features = ["age", "fare"]\nX = df[features]\ny = df["target"]',
            'X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)',
            'pipeline = Pipeline([])\npipeline.fit(X_train, y_train)',
            'pred = pipeline.predict(X_test)\naccuracy_score(y_test, pred)',
        ]
        for index, source in enumerate(sources, 1):
            notebook.cells.append(nbformat.v4.new_code_cell(source, execution_count=index))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "flow.ipynb"
            nbformat.write(notebook, path)
            flow = analyze_notebook_data_flow(path)
            issues = analyze_notebook(path)
        self.assertTrue(flow["split"]["detected"])
        self.assertEqual(flow["fit"][-1]["X"], "X_train")
        self.assertEqual(flow["predict"][-1]["X"], "X_test")
        self.assertEqual(flow["metric_coverage"]["detected"], ["accuracy"])
        rules = {issue["rule_id"] for item in issues for issue in item["issues"]}
        self.assertNotIn("CODE_NO_VALIDATION_SPLIT", rules)
        self.assertNotIn("CODE_NO_EVALUATION", rules)

    def test_04_dropna_impact_is_exact(self):
        frame = pd.DataFrame({
            "feature": list(range(150)) + [None] * 1850,
            "target": [0, 1] * 1000,
        })
        code = """
features = ["feature"]
X = df[features]
y = df["target"]
model_data = df[features + ["target"]]
model_data = model_data.dropna()
"""
        flow = analyze_python_data_flow(code)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            result = analyze_csv(path)
            flow = enrich_model_features(flow, result)
            impact = quantify_data_impacts(flow, path, result)[0]
        self.assertEqual(impact["quantification"], "Exact")
        self.assertEqual(impact["before"], 2000)
        self.assertEqual(impact["after"], 150)
        self.assertEqual(impact["removed"], 1850)
        self.assertAlmostEqual(impact["removal_rate"], 0.925)
        self.assertEqual(impact["relevant_columns"], ["feature"])

    def test_05_non_model_missing_column_is_excluded_from_model_impact(self):
        frame = pd.DataFrame({
            "Age": [20, None, 40],
            "Cabin": [None, None, "C1"],
            "Survived": [0, 1, 1],
        })
        code = """
features = ["Age"]
X = df[features]
y = df["Survived"]
model_data = df[features + ["Survived"]]
model_data = model_data.dropna()
"""
        flow = analyze_python_data_flow(code)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            frame.to_csv(path, index=False)
            result = analyze_csv(path)
            flow = enrich_model_features(flow, result)
            impact = quantify_data_impacts(flow, path, result)[0]
        self.assertEqual(flow["model_features"], ["Age"])
        self.assertNotIn("Cabin", impact["affected_columns"])
        self.assertNotIn("Cabin", impact["relevant_columns"])

    def test_06_negative_price_is_located_as_potential_business_violation(self):
        values = list(range(100, 130)) + [-20000, -1500, -500]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cars.csv"
            pd.DataFrame({"selling_price": values}).to_csv(path, index=False)
            result = analyze_csv(path)
        issue = next(item for item in result["issues"]
                     if item["rule_id"] == "DQ_POTENTIAL_NON_NEGATIVE_VIOLATION")
        self.assertEqual(issue["type"], "Potential Business Rule Violation")
        self.assertEqual(issue["confidence"], "Medium")
        self.assertEqual(issue["count"], 3)
        self.assertEqual(issue["affected_rows"], [30, 31, 32])
        self.assertEqual(set(issue["sample_values"]), {-20000, -1500, -500})

    def test_07_statistical_outlier_is_not_confirmed_invalid(self):
        incomes = list(range(1000, 1024)) + [1_000_000]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "income.csv"
            pd.DataFrame({"income": incomes}).to_csv(path, index=False)
            result = analyze_csv(path)
        outlier = next(item for item in result["issues"]
                       if item["rule_id"] == "DQ_POTENTIAL_EXTREME_VALUES")
        self.assertEqual(outlier["type"], "Potential Statistical Outlier")
        self.assertEqual(outlier["issue_nature"], "Statistical Anomaly")
        self.assertNotEqual(outlier["issue_nature"], "Confirmed Invalid")
        self.assertEqual(outlier["action"], "Review")

    def test_08_reproducible_split_detects_cross_split_duplicates(self):
        frame = pd.DataFrame({
            "feature": [value for value in range(10) for _ in (0, 1)],
            "target": [value % 2 for value in range(10) for _ in (0, 1)],
        })
        code = """
X = df[["feature"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)
"""
        flow = analyze_python_data_flow(code)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.csv"
            frame.to_csv(path, index=False)
            result = analyze_csv(path, data_flow=flow)
        split = result["duplicate_split_analysis"]
        self.assertEqual(split["status"], "Exact")
        self.assertGreater(split["cross_split_duplicate_groups"], 0)
        self.assertGreater(split["shared_records"], 0)
        self.assertIn("DQ_CROSS_SPLIT_DUPLICATES", {
            item["rule_id"] for item in result["issues"]
        })

    def test_09_accuracy_is_evaluation_but_coverage_is_limited(self):
        flow = analyze_python_data_flow(CLASSIFICATION_CODE)
        self.assertTrue(flow["evaluation_detected"])
        self.assertEqual(flow["metric_coverage"]["detected"], ["accuracy"])
        self.assertEqual(flow["metric_coverage"]["status"], "Metric Coverage Limited")

    def test_10_mae_is_evaluation_but_coverage_is_limited(self):
        code = """
model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
"""
        flow = analyze_python_data_flow(code)
        self.assertTrue(flow["evaluation_detected"])
        self.assertEqual(flow["task_type"]["type"], "Regression")
        self.assertEqual(flow["metric_coverage"]["detected"], ["mae"])
        self.assertEqual(flow["metric_coverage"]["status"], "Metric Coverage Limited")

    def test_11_repair_does_not_reference_excluded_cabin_and_is_syntax_valid(self):
        code = """
features = ["Age", "Fare"]
X = df[features]
y = df["Survived"]
model_data = df[features + ["Survived"]]
model_data = model_data.dropna()
"""
        flow = analyze_python_data_flow(code)
        code_result = [{"file": "model.py", "issues": analyze_code_with_ast(code)}]
        data_results = [{"file": "train.csv", "result": {
            "data_types": {"Age": "float64", "Fare": "float64", "Cabin": "object", "Survived": "int64"},
            "missing_values": {"Age": 0.2, "Cabin": 0.8},
        }}]
        suggestions = agent.build_repair_suggestions(code_result, flow, data_results)
        suggestions = validate_repair_suggestions(suggestions, flow, data_results)
        missing = next(item for item in suggestions if item["problem"] == "Missing Value Handling Risk")
        self.assertNotIn("Cabin", missing.get("after_code") or "")
        self.assertIn("Age", missing.get("after_code") or "")
        self.assertIn(missing["validation_status"], {"Fully Validated", "API Unverified", "Syntax Valid"})

    def test_issue_nature_does_not_mark_rare_categories_as_fix(self):
        values = ["common"] * 199 + ["rare"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "categories.csv"
            pd.DataFrame({"Pclass": values, "Fuel_Type": values}).to_csv(path, index=False)
            result = analyze_csv(path)
        rare = [item for item in result["issues"] if item["rule_id"] == "DQ_RARE_CATEGORIES"]
        self.assertTrue(rare)
        self.assertTrue(all(item["issue_nature"] == "Rare but Valid" for item in rare))
        self.assertTrue(all(item["action"] == "Informational" for item in rare))

    def test_12_consistency_validator_corrects_split_and_metric_contradictions(self):
        flow = analyze_python_data_flow(CLASSIFICATION_CODE)
        facts = build_analysis_facts([], [], flow)
        corrected, corrections = validate_agent_consistency(
            "No test set was created.\nNo evaluation was performed.", facts
        )
        self.assertNotIn("No test set", corrected)
        self.assertNotIn("No evaluation", corrected)
        self.assertIn("训练/测试划分", corrected)
        self.assertIn("accuracy", corrected)
        self.assertTrue(corrections)
        report = validate_report_consistency({
            "analysis_facts": facts,
            "ai_review": "No test set. No evaluation.",
            "code_quality": {
                "issue_count": 2,
                "issues": [
                    {"rule_id": "CODE_NO_VALIDATION_SPLIT"},
                    {"rule_id": "CODE_NO_EVALUATION"},
                ],
            },
        })
        self.assertEqual(report["code_quality"]["issues"], [])
        self.assertNotIn("No test set", report["ai_review"])


if __name__ == "__main__":
    unittest.main()
