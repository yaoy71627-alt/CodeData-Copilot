"""Final regression for result layers, artifact roles and bounded AI retries."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pandas as pd

import agent
from code_analyzer import analyze_code_with_ast
from consistency_validator import (build_analysis_facts, build_findings,
                                   group_findings, render_structured_review)
from data_analyzer import analyze_csv
from project_lineage import classify_dataset_role, apply_dataset_role_policy
from project_score import calculate_project_score
from report import _data_quality_score
from review_semantics import explanation_text_issue


def _entry(finding, **overrides):
    value = {
        "finding_id": finding["id"],
        "cause": "该规则记录了当前字段或代码路径中的可复核证据。",
        "potential_impact": "实际影响仍取决于模型输入与运行路径，需要对照验证。",
        "recommendation": "按原始证据核对当前对象，并记录验证结果。",
    }
    value.update(overrides)
    return value


def test_statistical_observations_are_retained_but_do_not_reduce_score(tmp_path):
    path = tmp_path / "clean.csv"
    values = list(range(100)) + [10_000]
    pd.DataFrame({"x": values, "y": [value * 2 for value in values]}).to_csv(path, index=False)
    result = analyze_csv(path)
    statistical = [issue for issue in result["issues"] if issue["rule_id"] in {
        "DQ_POTENTIAL_EXTREME_VALUES", "DQ_HIGH_CORRELATION",
    }]
    assert statistical
    assert all(issue["action"] == "Informational" and issue["score_impact"] == 0
               for issue in statistical)
    assert _data_quality_score(result) == 100
    findings = build_findings([], [{"file": path.name, "result": result}])
    assert all(finding["result_class"] == "informational_observation"
               for finding in findings if finding["rule_id"] in {
                   "DQ_POTENTIAL_EXTREME_VALUES", "DQ_HIGH_CORRELATION",
               })


def test_explicit_allowed_domain_is_not_overridden_by_dominant_distribution(tmp_path):
    path = tmp_path / "values.csv"
    pd.DataFrame({"Parch": [0] * 80 + [1] * 18 + [7] * 2}).to_csv(path, index=False)
    result = analyze_csv(path, constraints={"Parch": {"allowed_values": list(range(8))}})
    assert not any(issue["rule_id"] == "DQ_CONSTRAINT_VIOLATION" for issue in result["issues"])
    inferred = [issue for issue in result["issues"]
                if issue["rule_id"] == "DQ_POTENTIAL_CATEGORICAL_DOMAIN_VIOLATION"]
    assert all(issue["action"] == "Informational" for issue in inferred)


def test_result_csv_roles_use_artifact_specific_quality_policy():
    report = {"shape": {"rows": 5, "columns": 4}, "data_types": {
        "precision": "float64", "recall": "float64", "f1-score": "float64", "support": "int64",
    }, "issues": [
        {"rule_id": "DQ_VERY_SMALL_DATASET", "severity": "Low", "action": "Informational"},
        {"rule_id": "DQ_HIGH_CORRELATION", "severity": "Low", "action": "Review"},
    ]}
    role = classify_dataset_role("results/classification_report.csv", report, {"roles": ["unassociated"]})
    assert role["primary_role"] == "model_evaluation_report"
    apply_dataset_role_policy(report, role)
    assert all(issue["action"] == "Informational" and issue["score_impact"] == 0
               for issue in report["issues"])
    assert "训练集规模" in report["issues"][0]["recommendation"]

    importance = {"shape": {"rows": 20, "columns": 2},
                  "data_types": {"feature": "object", "importance": "float64"}}
    assert classify_dataset_role("results/feature_importance.csv", importance)["primary_role"] == "feature_importance_result"


def test_mae_and_r2_on_one_prediction_are_not_repeated_test_selection():
    source = """
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
"""
    issues = analyze_code_with_ast(source)
    assert not any(issue["rule_id"] == "CODE_TEST_SET_OVERUSE" for issue in issues)


def test_explanation_completeness_task_and_units_are_independently_testable():
    assert explanation_text_issue("tmp], ignore_index=True)")
    assert explanation_text_issue("建议通过，")
    assert explanation_text_issue("字段存在缺失值，建议核对处理路径。") is None

    code = [{"file": "model.py", "issues": [{
        "rule_id": "CODE_RISK", "type": "Model Review Risk", "line": 1,
        "severity": "Medium", "confidence": "Medium", "evidence": "分类模型输入需要复核。",
    }]}]
    facts = build_analysis_facts(code, [], {"task_type": {"type": "Classification", "confidence": "High"}})
    finding = facts["findings"][0]
    for bad in (
        _entry(finding, recommendation="建议改用 HuberRegressor 处理该分类任务。"),
        _entry(finding, cause="该字段以 kW 为单位且存在异常。"),
        _entry(finding, recommendation="tmp], ignore_index=True)"),
    ):
        review = render_structured_review({"explanations": [bad]}, facts)
        assert review.coverage["ai_explained"] == 0
        assert review.coverage["rejected_explanations"]


def test_retry_requests_only_missing_findings_and_preserves_first_success(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_RETRY_COUNT", "2")
    monkeypatch.setenv("DASHSCOPE_RETRY_BACKOFF", "0.05")
    code = [{"file": "model.py", "issues": [
        {"rule_id": "R1", "type": "Risk One", "line": 1, "severity": "High", "evidence": "first"},
        {"rule_id": "R2", "type": "Risk Two", "line": 2, "severity": "Medium", "evidence": "second"},
    ]}]
    facts = build_analysis_facts(code, [], {})
    ids = [finding["id"] for finding in facts["findings"]]
    calls = []

    def create(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        calls.append(prompt)
        requested = [identifier for identifier in ids if identifier in prompt]
        chosen = requested[:1] if len(calls) == 1 else requested
        payload = {"explanations": [_entry(next(f for f in facts["findings"] if f["id"] == fid))
                                    for fid in chosen]}
        return NS(choices=[NS(message=NS(content=json.dumps(payload, ensure_ascii=False)), finish_reason="stop")],
                  usage=NS(prompt_tokens=10, completion_tokens=10, total_tokens=20), _request_id=f"req-{len(calls)}")

    client = NS(chat=NS(completions=NS(create=Mock(side_effect=create))))
    with patch.object(agent, "get_client", return_value=client), patch.object(agent.time, "sleep"):
        review = agent.run_agent(code, [], knowledge_context="", analysis_facts=facts)
    assert review.ai_status["status"] == "success"
    assert len(calls) == 2 and sum(identifier in calls[1] for identifier in ids) == 1
    assert review.ai_status["batches"][0]["retry_count"] == 1
    assert review.ai_status["batches"][0]["success_count"] == 2
    assert set(review.coverage["accepted_annotations"]) == set(ids)


def test_duplicate_constant_column_evidence_counts_as_one_problem(tmp_path):
    path = tmp_path / "constant.csv"
    pd.DataFrame({"Over18": ["Y"] * 100}).to_csv(path, index=False)
    result = analyze_csv(path)
    findings = build_findings([], [{"file": path.name, "result": result}])
    constant = [finding for finding in findings if finding["rule_id"] in {
        "DQ_CONSTANT_COLUMN", "DQ_SINGLE_CATEGORY",
    }]
    assert len(constant) == 2
    assert len(group_findings(constant)) == 1
    score = calculate_project_score([], [{"file": path.name, "result": result}])
    assert score["breakdown"]["数据质量"] > 0
