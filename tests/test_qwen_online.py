"""Local opt-in only: tiny preflight, then a short real Titanic Notebook review."""
import os
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_online_preflight_then_short_titanic_same_configuration():
    from scripts.check_qwen_connection import check_connection
    from agent import run_agent
    from code_analyzer import analyze_notebook
    from consistency_validator import build_analysis_facts
    from data_flow_analyzer import analyze_notebook_data_flow
    path = Path(os.environ.get("CODEDATA_TITANIC_NOTEBOOK", ""))
    assert path.is_file(), "Set CODEDATA_TITANIC_NOTEBOOK to the original local problem Notebook"
    preflight = check_connection()
    assert preflight["status"] == "success", preflight
    blocks, remaining = [], 2
    for block in analyze_notebook(path):
        issues = [i for i in block["issues"] if i.get("severity") == "High"][:remaining]
        if issues:
            blocks.append({"file": path.name, **block, "issues": issues})
            remaining -= len(issues)
        if not remaining:
            break
    assert blocks, "The supplied Notebook did not yield the expected high-priority static findings"
    facts = build_analysis_facts(blocks, [], analyze_notebook_data_flow(path))
    review = run_agent(blocks, [], knowledge_context="", analysis_facts=facts)
    assert review.ai_status["status"] == "success", review.ai_status
