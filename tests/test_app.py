"""Smoke-test the V2 Streamlit product interface."""

import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAGES = (
    "Code Analysis",
    "Data Quality",
    "AI Review",
    "Repair",
    "Report",
)


def build_app(page="Home", *, with_results=False, import_mode=None):
    """Create a fresh AppTest because Streamlit 1.51 cannot mutate pills in place."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.session_state["active_page"] = page
    if import_mode:
        at.session_state["import_mode"] = import_mode

    if with_results:
        data_report = {
            "shape": {"rows": 100, "columns": 3},
            "missing_values": {"age": 0.1},
            "duplicate_rows": 2,
            "duplicate_rate": 0.02,
            "data_types": {"age": "float64", "income": "float64", "segment": "object"},
            "numeric_columns": ["age", "income"],
            "categorical_columns": ["segment"],
            "correlation": {
                "age": {"age": 1.0, "income": 0.4},
                "income": {"age": 0.4, "income": 1.0},
            },
        }
        at.session_state["analyzed_files"] = {
            "code": "1 个代码文件",
            "data": "1 个 CSV 文件",
            "documentation": "0 个 Markdown 文件",
        }
        at.session_state["project_stats"] = {
            "files": 1,
            "code_files": 1,
            "data_files": 1,
            "documentation_files": 0,
        }
        at.session_state["project_overview"] = {
            "project_name": "demo",
            "project_type": "Data Science Project",
            "summary": "静态项目摘要",
            "languages": ["Python"],
            "main_libraries": ["pandas"],
            "project_structure": ["数据读取与处理"],
            "data_files": [{
                "file": "demo.csv", "rows": 100, "columns": 3,
                "missing_columns": 1, "duplicate_rows": 2,
            }],
        }
        at.session_state["code_result"] = [{
            "file": "demo.py",
            "issues": [{
                "type": "Data Leakage Risk",
                "message": "fit_transform 可能在数据划分前拟合。",
                "source_snippet": "X_scaled = scaler.fit_transform(X)",
            }],
        }]
        at.session_state["data_result"] = data_report
        at.session_state["data_results"] = [{"file": "demo.csv", "result": data_report}]
        at.session_state["health"] = {
            "score": 35,
            "breakdown": {"代码质量": 35, "数据质量": 0, "项目完整性": 0},
            "explanation": [],
            "caveat": "静态评分",
        }

    return at.run()


class AppSmokeTests(unittest.TestCase):
    def test_home_and_top_navigation_render(self):
        at = build_app()
        self.assertFalse(at.exception)
        self.assertFalse(at.sidebar.radio)
        button_labels = [button.label for button in at.button]
        self.assertIn("Analyze Project", button_labels)
        self.assertIn("Code Review", button_labels)
        self.assertIn("Report", button_labels)
        self.assertNotIn("Workflow", button_labels)

    def test_pages_render_before_analysis(self):
        for page in PAGES:
            with self.subTest(page=page):
                at = build_app(page)
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state["active_page"], page)

    def test_github_import_route_renders(self):
        at = build_app(import_mode="GitHub Repository")
        self.assertFalse(at.exception)
        self.assertIn("GitHub repository", [item.label for item in at.text_input])
        self.assertIn("Analyze", [button.label for button in at.button])

    def test_local_launch_files_are_scoped_to_workspace(self):
        project = APP_PATH.parent
        launch = json.loads((project / ".vscode" / "launch.json").read_text(encoding="utf-8"))
        config = launch["configurations"][0]
        self.assertEqual(config["name"], "Run CodeData-Copilot")
        self.assertEqual(config["module"], "streamlit")
        self.assertEqual(config["args"], ["run", "app.py"])
        self.assertEqual(config["console"], "integratedTerminal")
        self.assertEqual(config["cwd"], "${workspaceFolder}")
        bat = (project / "run_app.bat").read_text(encoding="utf-8").lower()
        self.assertIn('cd /d "%~dp0"', bat)
        self.assertIn("python -m streamlit run app.py", bat)
        streamlit_config = (project / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn('address = "localhost"', streamlit_config)
        self.assertNotIn("0.0.0.0", streamlit_config)

    def test_result_pages_render_with_analysis_state(self):
        for page in PAGES:
            with self.subTest(page=page):
                at = build_app(page, with_results=True)
                self.assertFalse(at.exception)

if __name__ == "__main__":
    unittest.main()
