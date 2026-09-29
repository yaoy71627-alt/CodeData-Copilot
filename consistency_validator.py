"""Cross-check formal analyzer facts, AI narrative, repairs, and report output."""

from __future__ import annotations

from copy import deepcopy
import re


SPLIT_CONTRADICTIONS = (
    re.compile(r"\bno (?:explicit )?(?:train/?test split|test set|validation split)\b", re.I),
    re.compile(r"(?:没有|未发现|不存在).{0,12}(?:训练.?测试划分|测试集|train_test_split)", re.I),
)
EVALUATION_CONTRADICTIONS = (
    re.compile(r"\bno (?:explicit )?evaluation\b", re.I),
    re.compile(r"(?:没有|未发现|未进行|不存在).{0,12}(?:评价|评估|指标|metric)", re.I),
)
FAIRNESS_CONTRADICTIONS = (
    re.compile(r"\bno fairness (?:evaluation|metric|assessment)\b", re.I),
    re.compile(r"(?:没有|未发现|未进行|不存在).{0,10}(?:公平性评估|公平性指标|fairness)", re.I),
)
QUANTIFIED_CLAIM_RE = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>%|行|rows?)", re.I)


def build_analysis_facts(
    code_result,
    data_results,
    data_flow,
    quantified_impacts=None,
    duplicate_split_analysis=None,
    business_rule_results=None,
    artifact_analysis=None,
):
    """Build the sole structured fact bundle passed to AI and report renderers."""
    flow = data_flow or {}
    datasets = []
    for entry in data_results or []:
        result = entry.get("result", entry)
        datasets.append({
            "file": entry.get("file", "CSV"),
            "rows": result.get("shape", {}).get("rows", 0),
            "columns": result.get("shape", {}).get("columns", 0),
            "missing_values": result.get("missing_values", {}),
            "duplicate_rows": result.get("duplicate_rows", 0),
            "duplicate_rate": result.get("duplicate_rate", 0),
            "duplicate_groups": result.get("duplicate_groups", 0),
            "formal_data_issues": result.get("issues", []),
        })
    return {
        "detected_split": bool(flow.get("split", {}).get("detected")),
        "split": flow.get("split", {}),
        "model_features": list(flow.get("model_features", [])),
        "target": flow.get("target"),
        "fit": flow.get("fit", []),
        "predict": flow.get("predict", []),
        "detected_metrics": [item.get("label") for item in flow.get("metrics", [])],
        "metric_details": flow.get("metrics", []),
        "evaluation_detected": bool(flow.get("evaluation_detected")),
        "task_type": flow.get("task_type", {"type": "Unknown", "confidence": "Low"}),
        "metric_coverage": flow.get("metric_coverage", {}),
        "split_role_analysis": flow.get("split_role_analysis", {}),
        "fairness": flow.get("fairness", {}),
        "data_flow": flow,
        "quantified_impacts": list(quantified_impacts or []),
        "duplicate_split_analysis": duplicate_split_analysis or {},
        "business_rule_results": list(business_rule_results or []),
        "artifact_analysis": artifact_analysis or {"groups": [], "files": []},
        "formal_code_issues": [
            {**issue, "file": location.get("file"), "cell": location.get("cell")}
            for location in code_result or []
            for issue in location.get("issues", [])
        ],
        "datasets": datasets,
    }


def _allowed_quantities(facts):
    values = {"0", "1", "100"}
    for dataset in facts.get("datasets", []):
        values.update({str(dataset.get("rows", 0)), str(dataset.get("columns", 0)),
                       str(dataset.get("duplicate_rows", 0))})
        rate = float(dataset.get("duplicate_rate", 0)) * 100
        values.update({f"{rate:.1f}", f"{rate:.2f}"})
        for missing_rate in dataset.get("missing_values", {}).values():
            percent = float(missing_rate) * 100
            values.update({f"{percent:.1f}", f"{percent:.2f}"})
    for impact in facts.get("quantified_impacts", []):
        for key in ("before", "after", "removed"):
            if impact.get(key) is not None:
                values.add(str(impact[key]))
        if impact.get("removal_rate") is not None:
            percent = float(impact["removal_rate"]) * 100
            values.update({f"{percent:.1f}", f"{percent:.2f}"})
    for issue in facts.get("business_rule_results", []):
        if issue.get("count") is not None:
            values.add(str(issue["count"]))
        if issue.get("rate") is not None:
            percent = float(issue["rate"]) * 100
            values.update({f"{percent:.1f}", f"{percent:.2f}"})
    return values


def validate_agent_consistency(agent_text, facts):
    """Replace direct contradictions and ungrounded quantitative claims."""
    if not agent_text:
        return agent_text, []
    corrections = []
    metrics = [str(item) for item in facts.get("detected_metrics", []) if item]
    split_detected = bool(facts.get("detected_split"))
    evaluation_detected = bool(facts.get("evaluation_detected") or metrics)
    fairness = facts.get("fairness", {})
    allowed = _allowed_quantities(facts)
    output = []
    for line in str(agent_text).splitlines():
        corrected = line
        if split_detected and any(pattern.search(corrected) for pattern in SPLIT_CONTRADICTIONS):
            corrected = "已检测到训练/测试划分；该结论以静态数据流事实为准。"
            corrections.append("移除了与 detected_split=true 冲突的 AI 表述。")
        fairness_contradiction = any(pattern.search(corrected) for pattern in FAIRNESS_CONTRADICTIONS)
        if (evaluation_detected and not fairness_contradiction
                and any(pattern.search(corrected) for pattern in EVALUATION_CONTRADICTIONS)):
            metric_text = "、".join(metrics) or "predict/score"
            corrected = f"已检测到评价流程（{metric_text}）；评价覆盖是否充分需单独审查。"
            corrections.append("移除了与 detected_metrics 冲突的 AI 表述。")
        if fairness.get("evaluation_exists") and fairness_contradiction:
            corrected = (
                "已检测到公平性评估；数据集公平性指标与预测公平性指标按对象类型分别记录，"
                f"覆盖状态：{fairness.get('coverage', 'Unable to verify automatically.')}"
            )
            corrections.append("移除了与 fairness evaluation 事实冲突的 AI 表述。")
        claims = list(QUANTIFIED_CLAIM_RE.finditer(corrected))
        unsupported = [match.group("number") for match in claims if match.group("number") not in allowed]
        if unsupported and re.search(r"估算|估计|大约|约\s*\d|approximately|estimated", corrected, re.I):
            corrected = "Unable to verify automatically. 原 AI 数量估算未写入正式报告。"
            corrections.append("移除了未由 Analyzer 支持的 AI 数量估算。")
        output.append(corrected)
    return "\n".join(output), list(dict.fromkeys(corrections))


def validate_report_consistency(report):
    """Return a corrected report model before Markdown/PDF renderers consume it."""
    checked = deepcopy(report)
    facts = checked.get("analysis_facts", {})
    corrected_ai, corrections = validate_agent_consistency(checked.get("ai_review", ""), facts)
    checked["ai_review"] = corrected_ai

    if facts.get("detected_split"):
        before = len(checked.get("code_quality", {}).get("issues", []))
        checked["code_quality"]["issues"] = [
            issue for issue in checked.get("code_quality", {}).get("issues", [])
            if issue.get("rule_id") != "CODE_NO_VALIDATION_SPLIT"
        ]
        if len(checked["code_quality"]["issues"]) != before:
            corrections.append("移除了与项目级 split 事实冲突的 No Explicit Validation Split。")
    if facts.get("evaluation_detected"):
        before = len(checked.get("code_quality", {}).get("issues", []))
        checked["code_quality"]["issues"] = [
            issue for issue in checked.get("code_quality", {}).get("issues", [])
            if issue.get("rule_id") != "CODE_NO_EVALUATION"
        ]
        if len(checked["code_quality"]["issues"]) != before:
            corrections.append("移除了与项目级评价事实冲突的 No Explicit Evaluation。")
    checked["code_quality"]["issue_count"] = len(checked.get("code_quality", {}).get("issues", []))
    checked["consistency_corrections"] = list(dict.fromkeys(corrections))
    return checked
