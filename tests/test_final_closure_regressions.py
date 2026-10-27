"""Targeted regressions for the 2026-10-22 four-case closure."""
from io import BytesIO
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from pypdf import PdfReader

import agent
from code_analyzer import analyze_code_with_ast
from consistency_validator import build_analysis_facts, render_structured_review
from data_analyzer import analyze_csv
from data_flow_analyzer import analyze_python_data_flow, merge_data_flows
from project_score import calculate_project_score
from report import build_markdown_report, build_pdf_report, build_report_data
from review_semantics import explanation_text_issue


def _entry(finding, **overrides):
    item = {
        "finding_id": finding["id"],
        "cause": "该 Finding 的原始证据需要按当前对象复核。",
        "potential_impact": "实际影响取决于当前数据流，需要运行对照确认。",
        "recommendation": "按该 Finding 的原始证据核对当前对象并记录结果。",
    }
    item.update(overrides)
    return item


@pytest.fixture
def binding_facts():
    code = [{"file": "titanic.ipynb", "cell": 7, "issues": [{
        "rule_id": "CODE_TEST_SET_OVERUSE",
        "type": "Repeated Test-set Evaluation Requires Review",
        "category": "Evaluation",
        "line": 13,
        "severity": "Medium",
        "confidence": "Medium",
        "evidence": "测试集在循环中被重复评价；比较语义需要复核。",
        "recommendation": "使用验证集完成选择，仅对锁定方案评价测试集。",
    }]}]
    data = [{"file": "train.csv", "result": {
        "shape": {"rows": 916, "columns": 3},
        "data_types": {"Sex": "object", "Cabin": "object", "Fare": "float64"},
        "numeric_statistics": {"Fare": {"min": -100.0, "max": 512.0}},
        "issues": [{
            "rule_id": "DQ_MISSING_VALUES", "type": "Missing Values",
            "category": "Missing Values", "column": "Sex", "severity": "Medium",
            "confidence": "High", "evidence": "Sex 缺失 12/916 条记录。",
            "count": 12, "rate": 12 / 916,
            "recommendation": "结合 Sex 的缺失机制选择处理策略。",
        }, {
            "rule_id": "DQ_POTENTIAL_NON_NEGATIVE_VIOLATION",
            "type": "Potential Business Rule Violation", "category": "Business Rules",
            "column": "Fare", "severity": "Medium", "confidence": "Medium",
            "evidence": "Fare 最小值=-100；发现 4 个候选违规值。", "count": 4,
            "recommendation": "先确认非负约束，再处理违规值。",
        }],
    }}]
    return build_analysis_facts(code, data, {
        "task_type": {"type": "Classification", "confidence": "High"},
    })


def _finding(facts, rule):
    return next(item for item in facts["findings"] if item["rule_id"] == rule)


def test_titanic_sex_missing_cannot_receive_cabin_explanation(binding_facts):
    finding = _finding(binding_facts, "DQ_MISSING_VALUES")
    bad = _entry(
        finding,
        cause="Cabin 字段存在大量缺失且没有进入模型。",
        potential_impact="Cabin 缺失可能使模型输入不完整。",
        recommendation="为 Cabin 创建缺失指示变量并保留原始列。",
    )
    review = render_structured_review({"explanations": [bad]}, binding_facts)
    assert review.coverage["ai_explained"] == 0
    assert finding["id"] not in review.coverage["accepted_annotations"]
    assert all(bad[key] not in review for key in ("cause", "potential_impact", "recommendation"))
    assert "结构化事实补齐" in review


def test_test_evaluation_finding_cannot_receive_sex_missing_explanation(binding_facts):
    finding = _finding(binding_facts, "CODE_TEST_SET_OVERUSE")
    bad = _entry(
        finding,
        cause="Sex 字段存在缺失，当前处理不完整。",
        potential_impact="Sex 空值可能影响训练输入。",
        recommendation="生成 is_Sex_missing 特征后重新训练。",
    )
    review = render_structured_review({"explanations": [bad]}, binding_facts)
    assert review.coverage["ai_explained"] == 0
    assert "is_Sex_missing" not in review
    assert review.coverage["rejected_explanations"][0]["finding_id"] == finding["id"]


@pytest.mark.parametrize("field", ["potential_impact", "recommendation"])
def test_correct_cause_does_not_hide_misbound_impact_or_recommendation(binding_facts, field):
    finding = _finding(binding_facts, "DQ_MISSING_VALUES")
    values = {
        "cause": "Sex 字段缺失 12 条，与该 Finding 的证据一致。",
        "potential_impact": "Sex 是否进入实际模型决定其影响范围。",
        "recommendation": "针对 Sex 核对缺失机制并记录处理结果。",
    }
    values[field] = "Cabin 字段缺失，应为 Cabin 生成缺失指示变量。"
    review = render_structured_review({"explanations": [_entry(finding, **values)]}, binding_facts)
    assert review.coverage["ai_explained"] == 0
    assert values[field] not in review


@pytest.mark.parametrize("text", [
    "并保留原始列", "或为 train_df/test_df 提供兼容输入",
    "并验证下游稀疏矩阵行为是否一致",
])
def test_leading_conjunction_fragments_are_not_valid_ai_advice(text):
    assert explanation_text_issue(text)


def test_negative_fare_rejects_unconditional_log1p_but_allows_explicit_precondition(binding_facts):
    finding = _finding(binding_facts, "DQ_POTENTIAL_NON_NEGATIVE_VIOLATION")
    bad = _entry(finding, recommendation="直接对 Fare 使用 log1p 变换以减小偏斜。")
    rejected = render_structured_review({"explanations": [bad]}, binding_facts)
    assert rejected.coverage["ai_explained"] == 0
    assert "直接对 Fare 使用 log1p" not in rejected

    conditioned = _entry(
        finding,
        recommendation="先核实并处理 Fare 中小于 -1 的值，再对满足定义域的值使用 log1p。",
    )
    accepted = render_structured_review({"explanations": [conditioned]}, binding_facts)
    assert accepted.coverage["ai_explained"] == 1
    assert "小于 -1" in accepted and "使用 log1p" in accepted


def test_iqr_count_cannot_be_invented_from_bounds():
    data = [{"file": "train.csv", "result": {
        "shape": {"rows": 100, "columns": 1}, "data_types": {"Age": "float64"},
        "issues": [{
            "rule_id": "DQ_POTENTIAL_EXTREME_VALUES", "type": "Potential Statistical Outlier",
            "category": "Outliers", "column": "Age", "severity": "Low", "confidence": "Medium",
            "evidence": "方法=IQR；范围=[-5, 12]；min=-10，max=80。",
            "recommendation": "结合业务范围复核。",
        }],
    }}]
    facts = build_analysis_facts([], data, {})
    finding = facts["findings"][0]
    bad = _entry(finding, cause="IQR 区间外共有 12 条记录。")
    review = render_structured_review({"explanations": [bad]}, facts)
    assert review.coverage["ai_explained"] == 0
    assert bad["cause"] not in review


def test_same_prediction_mae_and_r2_do_not_create_split_role_risk_or_duplicate_summary():
    source = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
"""
    flow = analyze_python_data_flow(source, "usedcar.ipynb")
    assert flow["split_role_analysis"]["risks"] == []
    assert merge_data_flows([flow])["split_role_analysis"]["risks"] == []


def test_real_test_set_model_selection_is_still_detected():
    source = """
X_train, X_test, y_train, y_test = train_test_split(X, y)
for model in [LinearRegression(), RandomForestRegressor()]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = mean_absolute_error(y_test, pred)
"""
    issues = analyze_code_with_ast(source)
    assert any(item["rule_id"] == "CODE_TEST_SET_OVERUSE" for item in issues)
    risks = analyze_python_data_flow(source, "model.py")["split_role_analysis"]["risks"]
    assert [item["selection_type"] for item in risks if item["rule_id"] == "CODE_TEST_SET_OVERUSE"] == ["model_comparison"]


def test_informationals_never_penalize_and_high_value_refit_stays_visible():
    source = """
class PipelineLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return LabelEncoder().fit_transform(X)
"""
    high = next(item for item in analyze_code_with_ast(source)
                if item["rule_id"] == "CODE_TRANSFORMER_REFIT_DURING_TRANSFORM")
    informational = {
        "rule_id": "NB_SHELL_COMMAND", "type": "Shell Command", "category": "Notebook",
        "severity": "Low", "action": "Informational", "score_impact": 0,
    }
    baseline = calculate_project_score(
        [{"file": "encoder.py", "issues": [high]}], [], code_files_count=1,
    )
    repeated = calculate_project_score(
        [{"file": "encoder.py", "issues": [high, *([informational] * 100)]}],
        [], code_files_count=1,
    )
    assert repeated["score"] == baseline["score"]
    assert repeated["penalty_details"]["policy"]["informational_excluded"] is True
    assert repeated["penalty_details"]["code"]["sources"][0]["rule_ids"] == [
        "CODE_TRANSFORMER_REFIT_DURING_TRANSFORM"
    ]


@pytest.mark.parametrize("mode,expected", [
    ("all", "success"), ("partial", "partial_success"), ("none", "invalid_ai_response"),
])
def test_ai_success_states_follow_semantically_accepted_findings(monkeypatch, mode, expected):
    code = [{"file": "model.py", "issues": [
        {"rule_id": "R1", "type": "Risk One", "line": 1, "severity": "Medium", "evidence": "one"},
        {"rule_id": "R2", "type": "Risk Two", "line": 2, "severity": "Medium", "evidence": "two"},
    ]}]
    facts = build_analysis_facts(code, [], {})
    facts["datasets"] = [{"file": "train.csv", "column_names": ["Sex"]}]
    entries = [_entry(finding) for finding in facts["findings"]]
    if mode in {"partial", "none"}:
        start = 1 if mode == "partial" else 0
        for item in entries[start:]:
            item["recommendation"] = "为 Sex 缺失生成 is_Sex_missing 特征。"

    response = NS(
        choices=[NS(message=NS(content=json.dumps({"explanations": entries}, ensure_ascii=False)),
                    finish_reason="stop")],
        usage=NS(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    client = NS(chat=NS(completions=NS(create=Mock(return_value=response))))
    with patch.object(agent, "get_client", return_value=client), patch.object(agent.time, "sleep"):
        review = agent.run_agent(code, [], knowledge_context="", analysis_facts=facts)
    assert review.ai_status["status"] == expected
    assert review.ai_status["accepted_findings"] == {"all": 2, "partial": 1, "none": 0}[mode]


def test_explicit_enum_is_actionable_while_inferred_long_tail_remains_safe(tmp_path):
    path = tmp_path / "passengers.csv"
    pd.DataFrame({
        "Pclass": [1, 2, 3] * 32 + [4, 4, 4, 4],
        "Parch": [0, 1, 2, 3, 4, 5, 6, 7] * 12 + [0, 1, 2, 3],
    }).to_csv(path, index=False)
    result = analyze_csv(path, constraints={
        "source": "versioned schema v1",
        "Pclass": {"allowed_values": [1, 2, 3]},
        "Parch": {"allowed_values": list(range(8))},
    })
    violation = next(item for item in result["issues"]
                     if item["rule_id"] == "DQ_CONSTRAINT_VIOLATION")
    assert violation["column"] == "Pclass" and violation["count"] == 4
    assert violation["action"] == "Fix" and violation["confidence"] == "High"
    assert "versioned schema v1" in violation["issue_origin"]
    assert not any(item["column"] == "Parch" and item["rule_id"] == "DQ_CONSTRAINT_VIOLATION"
                   for item in result["issues"])
    assert not any(item["column"] == "Pclass" and item["rule_id"] == "DQ_POTENTIAL_CATEGORICAL_DOMAIN_VIOLATION"
                   for item in result["issues"])


def test_usedcar_style_counts_and_review_sources_match_markdown_pdf():
    observations = [{
        "rule_id": f"OBS_{index}", "type": f"Observation {index}",
        "category": "Statistics", "column": f"x{index}", "severity": "Low",
        "confidence": "Medium", "evidence": f"observation {index}",
        "recommendation": "保留为统计观察。", "action": "Informational", "score_impact": 0,
    } for index in range(4)]
    data = [{"file": "usedcar.csv", "result": {
        "shape": {"rows": 1400, "columns": 4},
        "data_types": {f"x{index}": "float64" for index in range(4)},
        "issues": observations,
    }}]
    facts = build_analysis_facts([], data, {})
    one = facts["findings"][0]
    review = render_structured_review({"explanations": [_entry(one)]}, facts)
    review.ai_status = {"status": "partial_success", "reason": "仅一项通过。"}
    report = build_report_data(
        {"project_name": "UsedCar", "file_count": 1, "python_file_count": 0, "data_file_count": 1},
        [], data, review, [], "", calculate_project_score([], data), analysis_facts=facts,
        enhancement_status={"ai": review.ai_status},
    )
    counts = report["conclusions"]["counts"]
    assert (counts["total"], counts["actionable_total"], counts["informational_observation"]) == (4, 0, 4)
    assert report["review_coverage"]["ai_explained"] == 1
    markdown = build_markdown_report(report)
    assert "需处理问题：0" in markdown and "信息性观察：4" in markdown
    assert "待人工复核：4" not in markdown and "质量问题：4" not in markdown
    assert "〔AI 解释〕" in markdown and "〔结构化事实补齐〕" in markdown

    pdf = build_pdf_report(report)
    pdf_text = "".join(page.extract_text() or "" for page in PdfReader(BytesIO(pdf)).pages)
    compact = lambda value: "".join(value.split())
    assert compact("需处理问题：0") in compact(pdf_text)
    assert compact("信息性观察：4") in compact(pdf_text)
    assert compact("〔AI 解释〕") in compact(pdf_text)
    assert compact("〔结构化事实补齐〕") in compact(pdf_text)
