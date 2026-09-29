"""Positive and false-positive regression tests for code and notebook review."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nbformat

from code_analyzer import analyze_code_with_ast, analyze_notebook, redact_sensitive_code


class CodeReviewRuleTests(unittest.TestCase):
    @staticmethod
    def rules(code):
        return {issue["rule_id"] for issue in analyze_code_with_ast(code)}

    def test_syntax_error_has_location_and_message(self):
        issue = analyze_code_with_ast("if True print('x')")[0]
        self.assertEqual(issue["rule_id"], "CODE_SYNTAX_ERROR")
        self.assertEqual(issue["severity"], "High")
        self.assertEqual(issue["line"], 1)
        self.assertTrue(issue["evidence"])

    def test_fit_transform_before_split_is_high_confidence_leakage(self):
        code = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
"""
        issue = next(item for item in analyze_code_with_ast(code)
                     if item["rule_id"] == "CODE_PREPROCESS_BEFORE_SPLIT")
        self.assertEqual((issue["severity"], issue["confidence"]), ("High", "High"))
        self.assertEqual(issue["line"], 5)

    def test_correct_split_then_fit_train_does_not_report_leakage(self):
        code = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
"""
        issues = analyze_code_with_ast(code)
        self.assertFalse([item for item in issues if item["category"] == "Data Leakage"])
        self.assertFalse([item for item in issues if item["severity"] in {"High", "Medium"}])

    def test_model_and_preprocessor_fit_on_test_data(self):
        model_rules = self.rules("model.fit(X_test, y_test)")
        self.assertIn("CODE_MODEL_FIT_TEST", model_rules)
        preprocess_rules = self.rules(
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\nscaler.fit(X_test)\n"
        )
        self.assertIn("CODE_PREPROCESSOR_FIT_TEST", preprocess_rules)

    def test_smote_before_split(self):
        code = """
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test = train_test_split(X_resampled, random_state=42)
"""
        rules = self.rules(code)
        self.assertIn("CODE_RESAMPLE_BEFORE_SPLIT", rules)
        self.assertNotIn("CODE_PREPROCESS_BEFORE_SPLIT", rules)

    def test_obvious_target_leakage(self):
        code = """
y = df["target"]
X = df
model.fit(X, y)
"""
        self.assertIn("CODE_TARGET_LEAKAGE", self.rules(code))
        clean = """
y = df["target"]
X = df.drop(columns=["target"])
model.fit(X, y)
"""
        self.assertNotIn("CODE_TARGET_LEAKAGE", self.rules(clean))

    def test_dropna_and_missing_random_state(self):
        code = """
from sklearn.model_selection import train_test_split
df = df.dropna()
X_train, X_test = train_test_split(X, y)
"""
        rules = self.rules(code)
        self.assertIn("CODE_DROPNA_SAMPLE_LOSS", rules)
        self.assertIn("CODE_SPLIT_RANDOM_STATE", rules)

    def test_hard_coded_path(self):
        issue = next(item for item in analyze_code_with_ast("df = pd.read_csv(r'C:\\data\\train.csv')")
                     if item["rule_id"] == "CODE_HARD_CODED_PATH")
        self.assertEqual(issue["confidence"], "High")

    def test_secret_is_detected_and_redacted_but_getenv_is_clean(self):
        secret = "sk-1234567890abcdef"
        issue = next(item for item in analyze_code_with_ast(f'api_key = "{secret}"')
                     if item["rule_id"] == "CODE_HARD_CODED_SECRET")
        self.assertNotIn(secret, issue["source_snippet"])
        self.assertNotIn(secret, redact_sensitive_code(f'api_key = "{secret}"'))
        self.assertNotIn("CODE_HARD_CODED_SECRET", self.rules('api_key = os.getenv("API_KEY")'))

    def test_secret_redaction_falls_back_when_source_has_syntax_error(self):
        secret = "sk-1234567890abcdef"
        redacted = redact_sensitive_code(f'api_key = "{secret}"\nif True print("broken")')
        self.assertNotIn(secret, redacted)
        self.assertIn("sk-...def", redacted)

    def test_dangerous_execution_rules_are_specific(self):
        rules = self.rules(
            "eval(payload)\nexec(script)\nos.system(command)\n"
            "subprocess.run(command, shell=True)\nsubprocess.run(['git', 'status'])\n"
        )
        self.assertTrue({"CODE_DANGEROUS_EVAL", "CODE_DANGEROUS_EXEC",
                         "CODE_OS_SYSTEM", "CODE_SUBPROCESS_SHELL"}.issubset(rules))

    def test_exception_mutable_default_and_wildcard_import(self):
        code = """
from package import *
def collect(items=[]):
    try:
        work()
    except:
        pass
"""
        rules = self.rules(code)
        self.assertIn("CODE_WILDCARD_IMPORT", rules)
        self.assertIn("CODE_MUTABLE_DEFAULT", rules)
        self.assertIn("CODE_SWALLOWED_EXCEPTION", rules)

    def test_pandas_iterrows_append_and_chained_assignment(self):
        code = """
import pandas as pd
df = pd.DataFrame({"x": [1]})
for _, row in df.iterrows():
    pass
df = df.append({"x": 2}, ignore_index=True)
df[df["x"] > 0]["y"] = 1
"""
        rules = self.rules(code)
        self.assertIn("CODE_PANDAS_ITERROWS", rules)
        self.assertIn("CODE_PANDAS_APPEND", rules)
        self.assertIn("CODE_CHAINED_ASSIGNMENT", rules)

    def test_training_only_evaluation(self):
        code = """
model.fit(X_train, y_train)
pred = model.predict(X_train)
accuracy_score(y_train, pred)
"""
        rules = self.rules(code)
        self.assertIn("CODE_TRAINING_ONLY_EVALUATION", rules)
        self.assertIn("CODE_SINGLE_CLASSIFICATION_METRIC", rules)

    def test_clean_pipeline_has_no_high_or_medium_findings(self):
        code = """
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(random_state=42)),
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
f1_score(y_test, pred)
cross_val_score(pipeline, X_train, y_train, cv=5)
"""
        issues = analyze_code_with_ast(code)
        self.assertFalse([item for item in issues if item["severity"] in {"High", "Medium"}])

    def test_notebook_out_of_order_execution_and_error_output(self):
        notebook = nbformat.v4.new_notebook()
        first = nbformat.v4.new_code_cell("x = 1", execution_count=3)
        second = nbformat.v4.new_code_cell("x + missing", execution_count=2)
        second.outputs = [nbformat.v4.new_output(
            output_type="error", ename="NameError", evalue="missing is not defined", traceback=[]
        )]
        notebook.cells = [first, second]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.ipynb"
            nbformat.write(notebook, path)
            results = analyze_notebook(path)
        rules = {issue["rule_id"] for result in results for issue in result["issues"]}
        self.assertIn("NB_NON_LINEAR_EXECUTION", rules)
        self.assertIn("NB_EXECUTION_ERROR", rules)


if __name__ == "__main__":
    unittest.main()
