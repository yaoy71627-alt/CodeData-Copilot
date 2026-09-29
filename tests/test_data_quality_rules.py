"""Focused positive and negative tests for the data quality analyzer."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from data_analyzer import analyze_csv, analyze_dataset_consistency


class DataQualityRuleTests(unittest.TestCase):
    def analyze_text(self, text, constraints=None):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text(text, encoding="utf-8")
            return analyze_csv(path, constraints=constraints)

    @staticmethod
    def rules(result):
        return {issue["rule_id"] for issue in result["issues"]}

    def test_missing_blank_tokens_and_duplicates(self):
        result = self.analyze_text(
            'value,token,space\n1,NA,"   "\n,ok,text\n1,NA,"   "\n'
        )
        rules = self.rules(result)
        self.assertIn("DQ_MISSING_VALUES", rules)
        self.assertIn("DQ_BLANK_STRING", rules)
        self.assertIn("DQ_SUSPECTED_MISSING_TOKEN", rules)
        self.assertIn("DQ_DUPLICATE_ROWS", rules)
        token_issue = next(item for item in result["issues"]
                           if item["rule_id"] == "DQ_SUSPECTED_MISSING_TOKEN")
        self.assertEqual(token_issue["confidence"], "Medium")

    def test_duplicate_columns_are_detected_from_raw_header(self):
        result = self.analyze_text("a,a,b\n1,2,3\n4,5,6\n")
        self.assertIn("DQ_DUPLICATE_COLUMN_NAME", self.rules(result))

    def test_constant_and_near_constant_columns(self):
        rows = ["constant,near"] + [f"x,{'a' if index < 99 else 'b'}" for index in range(100)]
        result = self.analyze_text("\n".join(rows))
        rules = self.rules(result)
        self.assertIn("DQ_CONSTANT_COLUMN", rules)
        self.assertIn("DQ_NEAR_CONSTANT_COLUMN", rules)

    def test_identical_columns_are_reported_once(self):
        result = self.analyze_text("left,right,other\n1,1,a\n2,2,b\n3,3,c\n")
        matches = [item for item in result["issues"]
                   if item["rule_id"] == "DQ_IDENTICAL_COLUMNS"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["severity"], "Medium")

    def test_positive_and_negative_infinity_are_non_finite(self):
        result = self.analyze_text("value\n1\ninf\n-inf\n4\n")
        issue = next(item for item in result["issues"] if item["rule_id"] == "DQ_NON_FINITE_NUMERIC")
        self.assertEqual(issue["confidence"], "High")
        self.assertEqual(issue["count"], 2)
        self.assertEqual(set(issue["sample_values"]), {"inf", "-inf"})

    def test_iqr_extreme_value_is_potential_not_invalid(self):
        values = list(range(1, 25)) + [1000]
        result = self.analyze_text("value\n" + "\n".join(map(str, values)))
        issue = next(item for item in result["issues"]
                     if item["rule_id"] == "DQ_POTENTIAL_EXTREME_VALUES")
        self.assertEqual(issue["confidence"], "Medium")
        self.assertIn("Potential", issue["type"])
        self.assertNotIn(issue, result["invalid_values"])

    def test_numeric_and_datetime_parsing_anomalies(self):
        numeric = [str(index) for index in range(1, 10)] + ["error"]
        dates = [f"2025-01-{index:02d}" for index in range(1, 10)] + ["bad-date"]
        text = "number,date\n" + "\n".join(f"{left},{right}" for left, right in zip(numeric, dates))
        result = self.analyze_text(text)
        rules = self.rules(result)
        self.assertIn("DQ_NUMERIC_PARSE_ANOMALY", rules)
        self.assertIn("DQ_DATETIME_PARSE_ANOMALY", rules)

    def test_mixed_numeric_and_text_representation_is_explicit(self):
        values = ["1", "2", "3", "4", "5", "north", "south", "east", "west", "other"]
        result = self.analyze_text("value\n" + "\n".join(values))
        issue = next(item for item in result["issues"]
                     if item["rule_id"] == "DQ_MIXED_REPRESENTATION")
        self.assertEqual(issue["confidence"], "Medium")
        self.assertIn("表示层诊断", issue["evidence"])

    def test_high_cardinality_and_potential_id(self):
        result = self.analyze_text(
            "user_id,feature\n" + "\n".join(f"user-{index},{index % 3}" for index in range(50))
        )
        rules = self.rules(result)
        self.assertIn("DQ_HIGH_CARDINALITY", rules)
        self.assertIn("DQ_POTENTIAL_ID", rules)

    def test_rare_category_requires_sufficient_sample(self):
        values = ["common"] * 199 + ["rare"]
        result = self.analyze_text("category\n" + "\n".join(values))
        self.assertIn("DQ_RARE_CATEGORIES", self.rules(result))

    def test_high_correlation_pair_is_not_duplicated(self):
        result = self.analyze_text(
            "a,b,c\n" + "\n".join(f"{index},{index * 2},{index % 7}" for index in range(50))
        )
        self.assertEqual(len(result["high_correlation_pairs"]), 1)
        self.assertIn("DQ_HIGH_CORRELATION", self.rules(result))

    def test_empty_dataset(self):
        result = self.analyze_text("")
        self.assertEqual(result["shape"], {"rows": 0, "columns": 0})
        self.assertIn("DQ_EMPTY_DATASET", self.rules(result))

    def test_explicit_constraint_enables_confirmed_violation(self):
        result = self.analyze_text("age\n20\n150\n", constraints={"age": {"min": 0, "max": 120}})
        issue = next(item for item in result["issues"] if item["rule_id"] == "DQ_CONSTRAINT_VIOLATION")
        self.assertEqual(issue["confidence"], "High")
        without_constraint = self.analyze_text("age\n20\n150\n")
        self.assertNotIn("DQ_CONSTRAINT_VIOLATION", self.rules(without_constraint))

    def test_potential_target_and_class_imbalance_are_medium_confidence(self):
        labels = ["major"] * 95 + ["minor"] * 5
        result = self.analyze_text("target\n" + "\n".join(labels))
        issue = next(item for item in result["issues"]
                     if item["rule_id"] == "DQ_POTENTIAL_CLASS_IMBALANCE")
        self.assertEqual(issue["confidence"], "Medium")

    def test_multi_csv_schema_and_dtype_differences(self):
        left = self.analyze_text("id,value\na,1\nb,2\n")
        right = self.analyze_text("ID,value,extra\na,text,x\nb,more,y\n")
        issues = analyze_dataset_consistency([
            {"file": "train.csv", "result": left},
            {"file": "test.csv", "result": right},
        ])
        rules = {item["rule_id"] for item in issues}
        self.assertIn("DQ_SCHEMA_DIFFERENCE", rules)
        self.assertIn("DQ_COLUMN_NAME_FORMAT", rules)
        self.assertIn("DQ_CROSS_FILE_DTYPE", rules)

    def test_clean_dataset_does_not_generate_high_or_many_medium_findings(self):
        frame = pd.DataFrame({
            "feature_a": list(range(60)),
            "feature_b": [(index * index + 3 * index) % 37 for index in range(60)],
            "segment": ["north", "south", "east"] * 20,
        })
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clean.csv"
            frame.to_csv(path, index=False)
            result = analyze_csv(path)
        high = [item for item in result["issues"] if item["severity"] == "High"]
        medium = [item for item in result["issues"] if item["severity"] == "Medium"]
        self.assertFalse(high)
        self.assertLessEqual(len(medium), 1)
        self.assertNotIn("DQ_NON_FINITE_NUMERIC", self.rules(result))
        self.assertNotIn("DQ_NUMERIC_PARSE_ANOMALY", self.rules(result))

    def test_every_issue_uses_the_stable_schema(self):
        result = self.analyze_text('value,token\n1,NA\n,ok\ninf,NA\n')
        required = {
            "rule_id", "category", "type", "severity", "confidence", "column",
            "message", "evidence", "count", "rate", "sample_values", "recommendation",
            "issue_origin", "issue_nature", "action", "affected_rows",
        }
        self.assertTrue(result["issues"])
        for issue in result["issues"]:
            self.assertTrue(required.issubset(issue), issue)


if __name__ == "__main__":
    unittest.main()
