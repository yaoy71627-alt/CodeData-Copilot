"""End-to-end reliability regression using a synthetic Titanic-equivalent fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import agent
from code_analyzer import analyze_code_with_ast
from consistency_validator import build_analysis_facts, validate_agent_consistency
from data_analyzer import analyze_csv
from data_flow_analyzer import (
    analyze_python_data_flow,
    quantify_data_impacts,
)
from repair_validator import validate_repair_suggestions
from report import build_markdown_report, build_pdf_report, build_report_data


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TITANIC_CODE = """
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
model_data = df[feature_cols + ["Survived"]].copy()
model_data = model_data.dropna()
X = model_data[feature_cols]
y = model_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
for model in [LogisticRegression(random_state=42), MLPClassifier(random_state=42)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
""".strip()


def _fixture_frame():
    rows = 916
    index = np.arange(rows)
    survived = index % 2
    frame = pd.DataFrame({
        "PassengerId": index + 1,
        "Survived": survived,
        "Pclass": (index % 3) + 1,
        "Name": [f"Passenger {value}" for value in index],
        "Sex": np.where(index % 2, "female", "male"),
        "Age": 18 + (index % 60) + index / 10_000,
        "SibSp": index % 4,
        "Parch": (index // 4) % 3,
        "Ticket": [str(value + 10_000) if value < 679 else f"PC-{value}" for value in index],
        "Fare": 20 + index / 100,
        "Cabin": [None if value % 3 else f"C{value}" for value in index],
        "Embarked": np.take(["S", "C", "Q"], index % 3),
    })
    frame.loc[[0, 1, 2, 3], "Pclass"] = [-1, 0, 4, 9]
    frame.loc[[4, 5, 6, 7], "Age"] = [-5, -1, -3, -7]
    frame.loc[[8, 9, 10, 11], "Fare"] = [-10, -50, -25, -100]
    frame.loc[647:, "Age"] = np.nan

    cleaned_index = list(range(647))
    train_index, test_index = train_test_split(
        cleaned_index,
        test_size=0.25,
        random_state=42,
        stratify=frame.loc[cleaned_index, "Survived"],
    )
    available_train = [value for value in train_index if value >= 100]
    available_test = [value for value in test_index if value >= 100]
    pairs = []
    used_test = set()
    for train_row in available_train:
        test_row = next(
            (
                value for value in available_test
                if value not in used_test
                and frame.at[value, "Survived"] == frame.at[train_row, "Survived"]
            ),
            None,
        )
        if test_row is None:
            continue
        pairs.append((train_row, test_row))
        used_test.add(test_row)
        if len(pairs) == 9:
            break
    for train_row, test_row in pairs:
        frame.loc[test_row, FEATURES] = frame.loc[train_row, FEATURES].to_numpy()
    return frame


class TitanicRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.csv_path = Path(cls.directory.name) / "titanic_equivalent.csv"
        _fixture_frame().to_csv(cls.csv_path, index=False)
        cls.flow = analyze_python_data_flow(TITANIC_CODE, "titanic.ipynb")
        cls.result = analyze_csv(cls.csv_path, data_flow=cls.flow)
        cls.code_issues = analyze_code_with_ast(TITANIC_CODE)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_a_model_features_and_dropna_impact(self):
        self.assertEqual(self.flow["model_features"], FEATURES)
        self.assertNotIn("Cabin", self.flow["model_features"])
        impacts = quantify_data_impacts(self.flow, self.csv_path, self.result)
        dropna = next(item for item in impacts if item["operation"] == "dropna")
        self.assertEqual((dropna["before"], dropna["after"], dropna["removed"]), (916, 647, 269))
        self.assertAlmostEqual(dropna["removal_rate"], 269 / 916)
        self.assertEqual(
            dropna["evidence_level"],
            "Verified from imported data + reconstructed operation",
        )

    def test_b_split_evaluation_and_model_comparison_semantics(self):
        modeling = self.result["duplicate_split_analysis"]["modeling_data"]
        self.assertEqual((modeling["train_rows"], modeling["test_rows"]), (485, 162))
        self.assertTrue(self.flow["evaluation_detected"])
        self.assertEqual(self.flow["metric_coverage"]["detected"], ["accuracy"])
        issue = next(item for item in self.code_issues if item["rule_id"] == "CODE_TEST_SET_OVERUSE")
        self.assertEqual(issue["type"], "Test-set Used for Model Comparison")
        self.assertEqual(issue["selection_type"], "model_comparison")
        self.assertNotIn("Hyperparameter", issue["type"])

    def test_c_cleaned_split_has_nine_duplicate_groups(self):
        duplicate = self.result["duplicate_split_analysis"]
        self.assertEqual(duplicate["status"], "Exact")
        self.assertEqual(duplicate["scope"], "model_input_row")
        self.assertEqual(duplicate["fields"], FEATURES)
        self.assertEqual(duplicate["cross_split_duplicate_groups"], 9)
        self.assertEqual(duplicate["modeling_data"]["cleaned_rows"], 647)

    def test_d_business_anomalies_and_ticket_identifier_handling(self):
        issues = self.result["issues"]
        inferred = {
            item["column"]: item
            for item in issues
            if item["rule_id"] == "DQ_POTENTIAL_NON_NEGATIVE_VIOLATION"
        }
        self.assertEqual(inferred["Fare"]["count"], 4)
        self.assertEqual(set(inferred["Fare"]["sample_values"]), {-10, -50, -25, -100})
        self.assertEqual(inferred["Age"]["count"], 4)
        self.assertEqual(set(inferred["Age"]["sample_values"]), {-5, -1, -3, -7})
        pclass = next(
            item for item in issues
            if item["rule_id"] == "DQ_POTENTIAL_CATEGORICAL_DOMAIN_VIOLATION"
            and item["column"] == "Pclass"
        )
        self.assertEqual(pclass["count"], 4)
        self.assertEqual(set(pclass["sample_values"]), {-1, 0, 4, 9})
        self.assertFalse(any(
            item["column"] == "Ticket"
            and item["rule_id"] in {"DQ_MIXED_REPRESENTATION", "DQ_NUMERIC_PARSE_ANOMALY"}
            for item in issues
        ))
        ticket = next(
            item for item in issues
            if item["column"] == "Ticket" and item["rule_id"] == "DQ_IDENTIFIER_REPRESENTATION"
        )
        self.assertEqual(ticket["action"], "Informational")

    def test_e_repair_context_rejects_undefined_fields_and_broken_downstream(self):
        code_result = [{"file": "titanic.ipynb", "issues": self.code_issues}]
        data_results = [{"file": self.csv_path.name, "result": self.result}]
        suggestions = validate_repair_suggestions(
            agent.build_repair_suggestions(code_result, self.flow, data_results),
            self.flow,
            data_results,
        )
        self.assertTrue(all("Cabin" not in (item.get("after_code") or "") for item in suggestions))
        overuse = next(item for item in suggestions if item["problem"] == "Test-set Used for Model Comparison")
        self.assertEqual(overuse["validation_status"], "Recommendation Only")

        dropna_line = TITANIC_CODE.splitlines().index("model_data = model_data.dropna()") + 1
        undefined = validate_repair_suggestions([{
            "file": "titanic.ipynb",
            "line": dropna_line,
            "before_code": "model_data = model_data.dropna()",
            "after_code": "imputer = SimpleImputer()\nX_train = imputer.fit_transform(X_train)",
        }], self.flow, data_results)[0]
        self.assertEqual(undefined["validation_status"], "Failed Validation")
        self.assertIn("X_train", undefined["validation_detail"])

        scaling_code = """X = df[["Age"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=1)
"""
        scaling_flow = analyze_python_data_flow(scaling_code, "scale.py")
        broken = validate_repair_suggestions([{
            "file": "scale.py", "line": 3,
            "before_code": "X_scaled = scaler.fit_transform(X)",
            "after_code": "scaler.fit(X)\nX_transformed = scaler.transform(X)",
        }], scaling_flow, data_results)[0]
        self.assertEqual(broken["validation_status"], "Failed Validation")
        self.assertIn("X_scaled", broken["validation_detail"])

    def test_f_grid_search_parameter_must_fit_every_candidate_estimator(self):
        flow = analyze_python_data_flow(TITANIC_CODE, "titanic.ipynb")
        suggestion = {
            "file": "titanic.ipynb",
            "line": 12,
            "before_code": "score = accuracy_score(y_test, y_pred)",
            "after_code": (
                "param_grid = {'C': [0.1, 1, 10]}\n"
                "search = GridSearchCV(model, param_grid=param_grid)\n"
                "search.fit(X_train, y_train)"
            ),
        }
        checked = validate_repair_suggestions(
            [suggestion], flow, [{"result": {"data_types": {name: "unknown" for name in FEATURES}}}],
        )[0]
        self.assertEqual(checked["validation_status"], "Failed Validation")
        self.assertIn("MLPClassifier", checked["validation_detail"])
        self.assertIn("C", checked["validation_detail"])

    def test_g_ai_wording_and_pdf_cjk_code_font(self):
        data_results = [{"file": self.csv_path.name, "result": self.result}]
        impacts = quantify_data_impacts(self.flow, self.csv_path, self.result)
        facts = build_analysis_facts(
            [], data_results, self.flow,
            quantified_impacts=impacts,
            duplicate_split_analysis=self.result["duplicate_split_analysis"],
            business_rule_results=self.result["business_rule_results"],
        )
        corrected, _ = validate_agent_consistency(
            "accuracy严重高估。\nSurvived=1是救命场景，应优先recall。", facts,
        )
        self.assertNotIn("严重高估", corrected)
        self.assertNotIn("救命场景", corrected)

        report = build_report_data(
            {
                "project_name": "Titanic fixture", "project_type": "Classification",
                "summary": "回归验证", "file_count": 2, "python_file_count": 1,
                "data_file_count": 1, "main_libraries": ["pandas", "scikit-learn"],
            },
            [],
            data_results,
            "## Review\n结论基于正式事实。",
            [{
                "problem": "示例", "location": "titanic.ipynb",
                "before_code": "# 中文注释：旧流程\nvalue = 1",
                "after_code": "# 中文注释：新流程\nvalue = 2",
                "validation_status": "Statically Validated",
            }],
            "",
            {"score": 90, "breakdown": {"代码质量": 35}, "caveat": "静态审计"},
            analysis_facts=facts,
        )
        pdf = build_pdf_report(report)
        markdown = build_markdown_report(report)
        self.assertIn("Before: 916 rows", markdown)
        self.assertIn("After: 647 rows", markdown)
        self.assertIn("Removed: 269 rows（29.37%）", markdown)
        self.assertIn("Train 485 / Test 162", markdown)
        self.assertIn("Cabin", markdown)
        self.assertIn("Cross-split duplicate groups (model_input_row): 9", markdown)
        self.assertTrue(pdf.startswith(b"%PDF"))
        # A named CID font alone does not guarantee readable Chinese.
        self.assertIn(b"/FontFile2", pdf)
        self.assertIn(b"NotoSansSC", pdf)


if __name__ == "__main__":
    unittest.main()
