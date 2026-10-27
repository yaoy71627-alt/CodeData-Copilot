"""Finding coverage is measured on one deduplicated problem-ID universe."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import agent
from consistency_validator import (
    ReviewText, build_analysis_facts, build_findings, build_review_coverage,
    coverage_for_review, group_findings, render_structured_review,
    validate_agent_consistency, validate_report_consistency,
)
from report import build_report_data, build_markdown_report, build_pdf_report


def fixture_findings():
    return build_findings([{"file": "titanic.py", "issues": [
        {"rule_id": "CODE_PREPROCESS_BEFORE_SPLIT", "type": "Data Leakage Risk", "line": 3,
         "severity": "High", "confidence": "High", "evidence": "scaler.fit_transform(X)",
         "recommendation": "先划分再拟合。", "category": "Leakage"},
        {"rule_id": "CODE_MISSING", "type": "Missing Handling Risk", "line": 8,
         "severity": "Medium", "confidence": "Medium", "evidence": "df.dropna()",
         "action": "Fix", "recommendation": "核查实际使用字段。", "category": "Missing Values"},
        {"rule_id": "CODE_OTHER", "type": "Other Risk", "line": 11,
         "severity": "Low", "confidence": "Low", "evidence": "Other static evidence",
         "category": "Reliability", "recommendation": "检查依赖。"},
    ]}], [])


def annotation(fid, **overrides):
    return {"finding_id": fid, "cause": "预处理拟合边界需与训练集一致。",
            "potential_impact": "若测试数据参与拟合，评价可能产生乐观偏差。",
            "recommendation": "请仅在训练集上拟合预处理器。", **overrides}


def test_important_omission_is_filled_once_without_model_retry():
    findings = fixture_findings()
    low = next(f for f in findings if f["severity"] == "Low")
    review = render_structured_review({"explanations": [annotation(low["id"])]}, {"findings": findings})
    coverage = review.coverage
    assert coverage["total"] == 3
    assert coverage["ai_explained"] == 1 and coverage["deterministic_filled"] == 2
    assert coverage["priority_total"] == 2 and coverage["priority_filled"] == 2
    for problem in coverage["items"]:
        assert problem["id"] in review
        for fid in problem["finding_ids"]:
            assert fid in review
        if problem["priority"]:
            assert problem["fallback"]
            assert problem["id"] in coverage["missing_problem_ids"]
    assert "CODE_PREPROCESS_BEFORE_SPLIT" in review
    assert "原因" in review and "潜在" in review and "结构化事实补齐" in review


@pytest.mark.parametrize("entry", [None, {}, {"cause": ""}, {"cause": "显著下降。"},
                                  {"cause": "有效原因。", "potential_impact": "", "recommendation": ""}])
def test_empty_partial_and_rejected_explanations_do_not_inflate_ai_count(entry):
    f = fixture_findings()[0]
    payload = {"explanations": [] if entry is None else [{"finding_id": f["id"], **entry}]}
    review = render_structured_review(payload, {"findings": [f]})
    assert review.coverage["ai_explained"] == 0
    assert review.coverage["deterministic_filled"] == 1
    assert review.coverage["manual_review"] == 1


def test_counts_are_subsets_on_same_ids_and_never_add_raw_occurrences():
    findings = fixture_findings()
    review = render_structured_review({"explanations": [annotation(f["id"]) for f in findings]}, {"findings": findings})
    c = review.coverage
    assert c["ai_explained"] + c["deterministic_filled"] == c["total"]
    assert c["manual_review"] == 3  # static risks need validation even if explained
    assert len({p["id"] for p in c["items"]}) == c["total"]
    assert "可与前两项重叠" in review


def test_cross_code_data_same_event_has_one_problem_and_preserves_both_findings():
    shared = {"deduplication_key": "dataset:sample.csv:invalid-age", "severity": "High", "confidence": "High"}
    findings = build_findings([{"file": "pipeline.py", "issues": [{**shared,
        "rule_id": "CODE_INVALID_INPUT", "type": "Invalid Input", "line": 7,
        "evidence": "The pipeline uses the invalid age field."}]}],
        [{"file": "sample.csv", "result": {"issues": [{**shared,
          "rule_id": "DQ_INVALID_AGE", "type": "Invalid Age", "column": "age", "count": 2,
          "evidence": "Explicit age constraint is violated."}]}}])
    review = render_structured_review({"explanations": [annotation(findings[0]["id"])]}, {"findings": findings})
    assert review.coverage["raw_finding_count"] == 2
    assert review.coverage["total"] == 1 and review.coverage["ai_explained"] == 1
    for f in findings:
        assert f["id"] in review and f["evidence"] in review
    assert len(review.coverage["items"][0]["missing_finding_ids"]) == 1


def test_identical_detection_repeated_in_input_is_not_multiple_errors():
    raw = {"rule_id": "DQ_EXAMPLE", "type": "Example", "severity": "Low", "evidence": "Same exact evidence"}
    fs = build_findings([], [{"file": "data.csv", "result": {"issues": [raw, copy.deepcopy(raw)]}}])
    assert len(fs) == 2 and fs[0]["id"] != fs[1]["id"]
    assert len(group_findings(fs)) == 1


def test_similar_but_different_locations_or_evidence_not_merged():
    fs = fixture_findings()
    other = copy.deepcopy(fs[0])
    other["id"] += "-other"
    other["location"]["line"] += 1
    other["evidence"] = "Other preprocessing event"
    assert len(group_findings([fs[0], other])) == 2


def test_same_rule_location_but_different_evidence_is_not_merged():
    fs = fixture_findings()
    second = copy.deepcopy(fs[0]); second["id"] += "-second"; second["evidence"] = "Different evidence"
    assert len(group_findings([fs[0], second])) == 2


def test_unknown_and_duplicate_ids_never_add_to_counts():
    fs = fixture_findings()
    review = render_structured_review({"explanations": [annotation("F-unknown"), annotation(fs[0]["id"]), annotation(fs[0]["id"])]}, {"findings": fs})
    assert review.coverage["total"] == 3
    assert review.coverage["ai_explained"] == 1


def test_priority_from_upstream_p0_or_confirmed_status_even_if_severity_low():
    fs = fixture_findings()
    fs[2]["severity"] = "Low"; fs[2]["priority"] = "P0"
    assert group_findings([fs[2]])[0]["priority"]
    fs[2]["priority"] = None; fs[2]["verification_status"] = "confirmed_error"
    assert group_findings([fs[2]])[0]["priority"]


def test_legacy_no_ai_reports_get_real_fallback_and_consistent_pdf_markdown():
    fs = fixture_findings()
    code = [{"file": f["location"]["file"], "issues": [{
        "rule_id": f["rule_id"], "type": f["type"], "severity": f["severity"],
        "confidence": f["confidence"], "line": f["location"]["line"], "evidence": f["evidence"],
        "recommendation": f["recommendation"], "action": f["action"],
    }]} for f in fs]
    report = build_report_data({"project_name": "Titanic"}, code, [], None, [], "", {}, analysis_facts={})
    report = validate_report_consistency(report)
    assert report["review_coverage"]["ai_explained"] == 0
    assert report["review_coverage"]["deterministic_filled"] == report["review_coverage"]["total"]
    markdown = build_markdown_report(report)
    assert "结构化事实补齐" in markdown and "检测问题总数：3" in markdown
    assert build_pdf_report(report).startswith(b"%PDF")


def test_metadata_survives_existing_app_validator_and_deepcopy_without_recounting_ids():
    fs = fixture_findings(); facts = {"findings": fs}
    review = render_structured_review({"explanations": [annotation(fs[0]["id"])]}, facts)
    checked, _ = validate_agent_consistency(review, facts)
    twice, _ = validate_agent_consistency(checked, facts)
    assert isinstance(twice, ReviewText)
    assert twice == checked and twice.coverage == checked.coverage == review.coverage
    assert copy.deepcopy(twice).coverage == twice.coverage


def test_all_important_findings_omitted_in_sdk_response_use_bounded_retries():
    code = [{"file": "titanic.py", "issues": [{"rule_id": "CODE_PREPROCESS_BEFORE_SPLIT", "type": "Data Leakage Risk", "severity": "High", "line": 3, "source_snippet": "scaler.fit_transform(X)"}]}]
    create = SimpleNamespace()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=create))
    with patch.object(create, "create", create=True, return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"explanations": []})))])) as call, patch.object(agent, "get_client", return_value=fake_client):
        review = agent.run_agent(code, [], knowledge_context="")
    from qwen_runtime import load_settings
    assert call.call_count == load_settings().retry_count + 1
    assert review.coverage["priority_filled"] == 1
    assert "scaler.fit_transform(X)" in review


def test_empty_findings_zero_counts_are_consistent():
    review = render_structured_review({}, {"findings": []})
    assert review.coverage["total"] == review.coverage["ai_explained"] == review.coverage["manual_review"] == 0


def test_coverage_recomputed_against_current_findings_not_stale_model_counts():
    fs = fixture_findings()
    review = render_structured_review({"explanations": [annotation(f["id"]) for f in fs]}, {"findings": fs})
    review.coverage["total"] = 9999
    assert coverage_for_review(review, fs)["total"] == 3
    assert coverage_for_review(review, [fs[0]])["total"] == 1


def test_stale_review_is_rebuilt_when_problem_universe_changes():
    from consistency_validator import ensure_review_coverage
    fs = fixture_findings()
    review = render_structured_review({"explanations": [annotation(f["id"]) for f in fs]}, {"findings": fs})
    updated = ensure_review_coverage(review, {"findings": [fs[0]]})
    assert updated.coverage["total"] == 1
    assert "检测问题总数：1" in updated and "检测问题总数：3" not in updated


def test_low_risks_share_concise_category_summary_without_losing_ids():
    raw = [{"rule_id": "DQ_SAME", "type": "Category Pattern", "severity": "Low", "confidence": "Medium",
            "column": column, "category": "Categorical Quality", "evidence": column,
            "recommendation": "核查类别编码。"} for column in ["first", "second"]]
    fs = build_findings([], [{"file": "data.csv", "result": {"issues": raw}}])
    review = render_structured_review({}, {"findings": fs})
    assert review.coverage["total"] == 2
    assert review.count("建议：核查类别编码。") == 1
    for f in fs:
        assert f["id"] in review


def test_report_statistics_survive_json_serializable_data_model():
    fs = fixture_findings()
    review = render_structured_review({"explanations": [annotation(fs[0]["id"])]}, {"findings": fs})
    model = {"ai_review": review, "review_coverage": review.coverage}
    restored = json.loads(json.dumps(model, ensure_ascii=False))
    assert restored["review_coverage"]["ai_explained"] == 1
    assert restored["review_coverage"]["total"] == 3


def test_legacy_dataset_issues_have_ids_and_share_existing_report_rules():
    from report import build_data_issues
    old = {"missing_values": {"age": 0.2, "fare": 0.1}, "duplicate_rows": 2, "duplicate_rate": 0.02}
    findings = build_findings([], [{"file": "legacy.csv", "result": old}])
    assert len(findings) == len(build_data_issues(old)) == 3
    assert {f["rule_id"] for f in findings} == {"DQ_MISSING_VALUES", "DQ_DUPLICATE_ROWS"}
    review = render_structured_review({}, {"findings": findings})
    assert review.coverage["total"] == 3
    for f in findings:
        assert f["id"] in review
