"""Integration checks for the quality-review data contract and downstream consumers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agent
from code_analyzer import analyze_code_with_ast
from data_analyzer import analyze_csv
from project_score import calculate_project_score
from report import build_markdown_report, build_report_data


class QualityIntegrationTests(unittest.TestCase):
    def test_code_findings_use_the_stable_schema(self):
        issues = analyze_code_with_ast(
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X = scaler.fit_transform(X)\n"
        )
        required = {
            "rule_id", "category", "type", "severity", "confidence", "message",
            "recommendation", "line", "end_line", "source_snippet",
        }
        self.assertTrue(issues)
        for issue in issues:
            self.assertTrue(required.issubset(issue), issue)

    def test_score_uses_severity_and_caps_repeated_rules(self):
        def code_issue(severity):
            return {
                "rule_id": "CODE_EXAMPLE", "category": "Reliability",
                "type": "Example", "severity": severity,
            }

        low = calculate_project_score(
            [{"file": "model.py", "issues": [code_issue("Low")]}],
            [{"shape": {"rows": 20, "columns": 2}, "issues": []}],
            code_files_count=1,
        )
        high = calculate_project_score(
            [{"file": "model.py", "issues": [code_issue("High")]}],
            [{"shape": {"rows": 20, "columns": 2}, "issues": []}],
            code_files_count=1,
        )
        repeated = calculate_project_score(
            [{"file": "model.py", "issues": [code_issue("High")] * 100}],
            [{"shape": {"rows": 20, "columns": 2}, "issues": []}],
            code_files_count=1,
        )
        self.assertLess(high["score"], low["score"])
        self.assertGreaterEqual(repeated["breakdown"]["代码质量"], 23)
        self.assertIn("启发式静态评分", high["caveat"])

    def test_new_findings_reach_knowledge_and_markdown_report(self):
        code_result = [{
            "file": "model.py",
            "issues": analyze_code_with_ast(
                'api_key = "sk-1234567890abcdef"\n'
                "value = eval(payload)\n"
            ),
        }]
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "train.csv"
            csv_path.write_text(
                "value,copy\n" + "\n".join(
                    [f"{number},{number}" for number in range(1, 18)]
                    + ["1000,1000", "1100,1100", "1200,1200"]
                ),
                encoding="utf-8",
            )
            data_result = analyze_csv(csv_path)

        knowledge = agent.retrieve_relevant_knowledge(
            code_result,
            [{"file": "train.csv", "result": data_result}],
            max_items=20,
        )
        self.assertIn("Security and Reliability Review", knowledge)
        self.assertIn("Outliers, Invalid Values and Type Consistency", knowledge)

        health = calculate_project_score(
            code_result,
            [{"file": "train.csv", "result": data_result}],
            code_files_count=1,
        )
        report_data = build_report_data(
            {
                "project_name": "integration",
                "project_type": "Data Science Project",
                "summary": "Static integration check",
                "file_count": 2,
                "python_file_count": 1,
                "data_file_count": 1,
                "main_libraries": [],
            },
            code_result,
            [{"file": "train.csv", "result": data_result}],
            "Static review only.",
            [],
            "",
            health,
        )
        markdown = build_markdown_report(report_data)
        self.assertIn("Hard-coded Credential", markdown)
        self.assertIn("置信度", markdown)
        self.assertIn("Potential Statistical Outlier", markdown)
        self.assertIn("IQR", markdown)
        self.assertNotIn("sk-1234567890abcdef", markdown)


if __name__ == "__main__":
    unittest.main()
