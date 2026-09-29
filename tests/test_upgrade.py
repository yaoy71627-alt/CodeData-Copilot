"""Offline regression checks for the project upgrade."""

import unittest
from pathlib import Path
from subprocess import TimeoutExpired
from types import SimpleNamespace
from unittest.mock import patch

import agent
import github_loader
from github_loader import GitHubImportError, load_github_project, validate_github_url
from code_analyzer import analyze_notebook
from data_analyzer import analyze_csv
from patch_generator import build_unified_patch
from project_overview import build_project_overview
from project_score import calculate_project_score


class UpgradeTests(unittest.TestCase):
    def test_github_url_is_restricted_to_repository_root(self):
        self.assertEqual(validate_github_url("https://github.com/acme/demo"),
                         "https://github.com/acme/demo.git")
        for url in ("http://github.com/a/b", "https://evil.example/a/b",
                    "https://github.com/a/b/tree/main", "https://github.com/a/b?token=x",
                    "https://github.com/a/b@evil.example/c"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_github_url(url)

    def test_import_scans_supported_files_and_cleans_up(self):
        def fake_clone(args, **kwargs):
            root = Path(args[-1])
            root.mkdir()
            (root / "model.py").write_text("model.fit(X, y)", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with patch("github_loader.run", side_effect=fake_clone):
            with load_github_project("https://github.com/acme/demo") as files:
                self.assertEqual([name for name, _ in files], ["model.py"])
                clone_root = files[0][1].parent
        self.assertFalse(clone_root.exists())

    def test_import_records_skips_and_supports_markdown(self):
        def fake_clone(args, **kwargs):
            root = Path(args[-1])
            root.mkdir()
            (root / "README.md").write_text("# Demo", encoding="utf-8")
            (root / "model.py").write_text("model.fit(X, y)", encoding="utf-8")
            (root / "weights.png").write_bytes(b"binary")
            (root / "large.csv").write_bytes(b"a" * 200)
            (root / "models").mkdir()
            (root / "models" / "ignored.py").write_text("pass", encoding="utf-8")
            return SimpleNamespace(returncode=0, stderr="")

        with patch("github_loader.run", side_effect=fake_clone), \
                patch.object(github_loader, "MAX_FILE_SIZE", 50):
            with load_github_project("https://github.com/acme/demo") as files:
                self.assertEqual([name for name, _ in files], ["README.md", "model.py"])
                skipped = {item["path"]: item["reason"] for item in files.skipped_files}
        self.assertIn("weights.png", skipped)
        self.assertIn("large.csv", skipped)
        self.assertIn("models/", skipped)
        self.assertIn("超过", skipped["large.csv"])

    def test_clone_timeout_is_detailed_and_uses_300_seconds(self):
        captured = {}

        def fake_clone(*args, **kwargs):
            captured.update(kwargs)
            raise TimeoutExpired(cmd="git clone", timeout=kwargs["timeout"])

        with patch("github_loader.run", side_effect=fake_clone):
            with self.assertRaises(GitHubImportError) as caught:
                with load_github_project("https://github.com/acme/demo"):
                    pass
        self.assertEqual(captured["timeout"], 300)
        self.assertEqual(caught.exception.category, "clone_timeout")
        self.assertIn("300", str(caught.exception))

    def test_clone_errors_distinguish_repository_and_network(self):
        cases = (
            ("fatal: repository not found", "invalid_repository"),
            ("fatal: could not resolve host: github.com", "network_problem"),
        )
        for stderr, category in cases:
            with self.subTest(category=category), patch(
                    "github_loader.run",
                    return_value=SimpleNamespace(returncode=128, stderr=stderr)):
                with self.assertRaises(GitHubImportError) as caught:
                    with load_github_project("https://github.com/acme/demo"):
                        pass
                self.assertEqual(caught.exception.category, category)

    def test_project_score_is_explainable_without_workflow(self):
        score = calculate_project_score([], [{"shape": {"rows": 10},
                                               "missing_values": {}, "duplicate_rate": 0}],
                                        code_files_count=1)
        self.assertEqual(score["score"], 100)
        self.assertEqual(score["breakdown"]["项目完整性"], 30)
        self.assertEqual(calculate_project_score([], [], code_files_count=0)["score"], 0)

    def test_existing_demo_analyzers_remain_compatible(self):
        project = Path(__file__).resolve().parents[1]
        code = analyze_notebook(str(project / "demo_data" / "test.ipynb"))
        data = analyze_csv(str(project / "demo_data" / "train.csv"))
        self.assertIsInstance(code, list)
        self.assertGreater(data["shape"]["rows"], 0)
        self.assertIn("score", calculate_project_score(code, [data], code_files_count=1))

    def test_agent_uses_knowledge_and_real_snippets(self):
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="修复建议"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
        with patch.object(agent, "get_client", return_value=client):
            result = agent.run_agent(
                [{"issues": [{"type": "Data Leakage Risk",
                              "source_snippet": "scaler.fit_transform(X)"}]}],
                [],
                {"project_files": ["model.py"], "user_code_context": "model training"},
            )
        self.assertEqual(result, "修复建议")
        self.assertIn("scaler.fit_transform(X)", captured["messages"][1]["content"])
        self.assertIn("Data Leakage Prevention", captured["messages"][1]["content"])
        self.assertIn("model.py", captured["messages"][1]["content"])
        for heading in ("## 1. 项目总结", "## 2. 发现的问题", "## 3. 问题原因",
                        "## 4. 影响分析", "## 5. 优化建议", "## 6. 修改代码示例"):
            self.assertIn(heading, captured["messages"][1]["content"])

    def test_offline_repair_never_invents_original_code(self):
        suggestions = agent.build_repair_suggestions([
            {"file": "model.py", "issues": [{"type": "Data Leakage Risk"}]}
        ])
        self.assertEqual(len(suggestions), 1)
        self.assertIsNone(suggestions[0]["original_code"])
        self.assertIn("X_test = scaler.transform(X_test)", suggestions[0]["corrected_code"])
        self.assertTrue(suggestions[0]["requires_confirmation"])

    def test_patch_export_is_unified_diff_and_never_edits_source(self):
        source = "scaler.fit_transform(X)"
        suggestions = agent.build_repair_suggestions([
            {"file": "model.py", "issues": [{
                "type": "Data Leakage Risk", "source_snippet": source,
            }]}
        ])
        patch_text = build_unified_patch(suggestions)
        self.assertIn("--- a/model.py", patch_text)
        self.assertIn("+++ b/model.py", patch_text)
        self.assertIn("-scaler.fit_transform(X)", patch_text)
        self.assertEqual(source, "scaler.fit_transform(X)")

    def test_project_overview_detects_libraries_and_project_structure(self):
        overview = build_project_overview(
            ["README.md", "train.py", "train.csv"],
            ["import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
             "df = pd.read_csv('train.csv')\nmodel.fit(X, y)\nmodel.predict(X)"],
            [{"file": "train.csv", "result": {
                "shape": {"rows": 10, "columns": 3}, "missing_values": {}, "duplicate_rows": 0,
            }}],
            {"repository_url": "https://github.com/acme/auditor.git"},
        )
        self.assertEqual(overview["project_name"], "auditor")
        self.assertIn("pandas", overview["main_libraries"])
        self.assertIn("scikit-learn", overview["main_libraries"])
        self.assertIn("模型构建与训练", overview["project_structure"])
        self.assertEqual(overview["data_files"][0]["rows"], 10)

    def test_v2_knowledge_files_exist(self):
        knowledge = Path(__file__).resolve().parents[1] / "knowledge"
        expected = {"data_quality.md", "code_quality.md", "data_leakage.md",
                    "feature_engineering.md", "model_evaluation.md",
                    "reproducibility.md", "notebook_quality.md",
                    "sklearn_pandas_best_practice.md", "outliers_invalid_values.md",
                    "security_reliability.md", "data_science_code_review.md"}
        self.assertTrue(expected.issubset({path.name for path in knowledge.glob("*.md")}))

    def test_full_demo_pipeline_without_network(self):
        import app

        project = Path(__file__).resolve().parents[1]
        state = SimpleNamespace()
        with patch.object(app.st, "session_state", state):
            app.analyze_project_files([
                ("test.ipynb", project / "demo_data" / "test.ipynb"),
                ("train.csv", project / "demo_data" / "train.csv"),
            ], generate_ai=False)
        self.assertEqual(len(state.data_results), 1)
        self.assertTrue(0 <= state.health["score"] <= 100)
        self.assertFalse(hasattr(state, "workflow"))
        self.assertIn("project_name", state.project_overview)
        self.assertIsNone(state.agent_report)

    def test_local_upload_pipeline_materializes_and_analyzes_files(self):
        import app

        project = Path(__file__).resolve().parents[1]

        class Uploaded:
            def __init__(self, path):
                self.name = path.name
                self._content = path.read_bytes()

            def getvalue(self):
                return self._content

        state = SimpleNamespace()
        with patch.object(app.st, "session_state", state):
            app.run_full_analysis(
                Uploaded(project / "demo_data" / "test.ipynb"),
                Uploaded(project / "demo_data" / "train.csv"),
                generate_ai=False,
            )
        self.assertEqual(state.import_source, "Local upload")
        self.assertEqual(state.project_stats["code_files"], 1)
        self.assertEqual(state.project_stats["data_files"], 1)
        self.assertGreater(state.health["score"], 0)

    def test_multi_file_local_project_upload(self):
        import app

        project = Path(__file__).resolve().parents[1]

        class Uploaded:
            def __init__(self, path):
                self.name = path.name
                self._content = path.read_bytes()

            def getvalue(self):
                return self._content

        state = SimpleNamespace()
        uploads = [Uploaded(project / "demo_data" / "test.ipynb"),
                   Uploaded(project / "demo_data" / "train.csv")]
        with patch.object(app.st, "session_state", state):
            app.run_uploaded_project(uploads, generate_ai=False)
        self.assertEqual(state.project_overview["file_count"], 2)
        self.assertEqual(state.import_source, "Local upload")

    def test_py_ipynb_csv_pipeline_reports_progress_and_exports(self):
        import app

        project = Path(__file__).resolve().parents[1]

        class Uploaded:
            def __init__(self, name, content):
                self.name = name
                self._content = content

            def getvalue(self):
                return self._content

        uploads = [
            Uploaded("model.py", b"X_scaled = scaler.fit_transform(X)\n"),
            Uploaded("test.ipynb", (project / "demo_data" / "test.ipynb").read_bytes()),
            Uploaded("train.csv", (project / "demo_data" / "train.csv").read_bytes()),
        ]
        events = []
        state = SimpleNamespace()
        with patch.object(app.st, "session_state", state):
            app.run_uploaded_project(
                uploads,
                generate_ai=False,
                progress_callback=lambda index, status, detail: events.append(
                    (index, status, detail)
                ),
            )

        completed = {index for index, status, _ in events if status == "complete"}
        self.assertEqual(completed, set(range(7)))
        self.assertEqual(state.project_stats["code_files"], 2)
        self.assertEqual(state.project_stats["data_files"], 1)
        self.assertIn("# CodeData-Copilot Analysis Report", state.report_markdown)
        self.assertIn("## 5. Repair Suggestions", state.report_markdown)
        self.assertIsNotNone(state.report_pdf, state.report_error)
        self.assertTrue(state.report_pdf.startswith(b"%PDF"))
        self.assertIn("--- a/model.py", state.patch_text)
        self.assertFalse(state.report_error)


if __name__ == "__main__":
    unittest.main()
