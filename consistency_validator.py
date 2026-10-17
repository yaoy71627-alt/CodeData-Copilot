"""Cross-check formal analyzer facts, AI narrative, repairs, and report output."""

from __future__ import annotations

from copy import deepcopy
import re
import json
import hashlib


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
QUANTIFIED_CLAIM_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>%|rows?|groups?|records?|行|条|组|个)",
    re.I,
)
UNSUPPORTED_IMPACT_RE = re.compile(
    r"(?:accuracy|准确率|精度)?.{0,10}(?:严重高估|显著高估|显著提升|显著降低|大幅偏差|严重降低)",
    re.I,
)
UNSUPPORTED_BUSINESS_CONTEXT_RE = re.compile(
    r"(?:救命场景|救命|医疗风险|金融风险|业务损失)", re.I,
)



STATUS_LABELS = {
    "confirmed_error": "已证实错误", "static_risk": "静态分析风险",
    "needs_review": "需要人工复核",
}



class ReviewText(str):
    """Keep typed coverage metadata alongside the legacy Markdown string API."""
    def __new__(cls, value, coverage=None):
        instance = super().__new__(cls, value)
        instance.coverage = deepcopy(coverage or {})
        return instance


def is_priority_finding(finding):
    return (
        str(finding.get("severity", "")).casefold() in {"high", "critical"}
        or str(finding.get("priority", "")).casefold() in {"p0", "p1", "urgent", "high"}
        or finding.get("action") == "Fix"
        or finding.get("verification_status") == "confirmed_error"
    )


def group_findings(findings):
    """Deduplicate proven identical events; never merge merely similar risk names."""
    grouped = {}
    for finding in sorted(findings, key=lambda item: item["id"]):
        explicit = finding.get("deduplication_key")
        loc = finding["location"]
        # Rule, scope and exact evidence must agree unless the analyzer explicitly
        # supplied a common event key across code/data rules.
        identity = (["event", explicit] if explicit else [
            "observation", finding.get("rule_id") or finding["type"], loc,
            finding.get("evidence"), finding.get("source_snippet"),
        ])
        key = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
        pid = "P-" + hashlib.sha256(key.encode()).hexdigest()[:16]
        grouped.setdefault(pid, []).append(finding)
    problems = []
    for pid, members in sorted(grouped.items()):
        problems.append({
            "id": pid, "finding_ids": [f["id"] for f in members],
            "priority": any(is_priority_finding(f) for f in members),
            "category": members[0].get("category") or members[0]["type"],
            "needs_verification": any(f["verification_status"] != "confirmed_error" for f in members),
            "deduplication_reason": "底层明确给出相同事件标识" if members[0].get("deduplication_key") else "规则、位置和证据完全相同",
        })
    return problems


def deterministic_explanation(finding):
    """Use only a known rule and its evidence boundary; never infer runtime harm."""
    rule = str(finding.get("rule_id") or "未提供规则编号")
    label = STATUS_LABELS.get(finding["verification_status"], finding["verification_status"])
    cause = f"底层规则 {rule} 记录了“{finding['type']}”；证据见对应 Finding，当前验证状态为{label}。"
    if "FEATURE_COMBINATIONS" in rule:
        impact = "相同特征组合不等于相同业务实体；是否影响评估，需要业务主键核查和运行对照。"
    elif "LEAK" in rule or "PREPROCESS" in rule or "leakage" in finding["type"].casefold():
        impact = "若拟合或划分使用了测试集信息，评价可能产生乐观偏差；当前不能确认性能变化幅度。"
    elif finding.get("category") == "Missing Values":
        impact = "缺失值是否影响模型输入取决于实际使用字段和处理路径，需要结合数据流复核。"
    else:
        impact = "请核对该证据对实际执行路径或数据使用的影响；静态发现不能证明性能损失。"
    return {"cause": cause, "potential_impact": impact,
            "recommendation": finding["recommendation"]}


def build_review_coverage(findings, annotations=None):
    """All counts use the same deduplicated problem-ID universe, with subset IDs."""
    annotations = annotations or {}
    problems = group_findings(findings)
    items = []
    for problem in problems:
        ids = problem["finding_ids"]
        complete = [fid for fid in ids if all(annotations.get(fid, {}).get(k) for k in
                                             ("cause", "potential_impact", "recommendation"))]
        partial = [fid for fid in ids if any(annotations.get(fid, {}).values())]
        items.append({**problem,
            "ai_explained": bool(complete),
            "ai_explained_finding_ids": complete,
            "mentioned_finding_ids": partial,
            "missing_finding_ids": [fid for fid in ids if fid not in partial],
            "fallback": not bool(complete),
            "manual_review": problem["needs_verification"] or not bool(complete),
        })
    return {
        "schema_version": 1, "counting_unit": "deduplicated_problem_id",
        "total": len(items), "raw_finding_count": len(findings),
        "ai_explained": sum(item["ai_explained"] for item in items),
        "deterministic_filled": sum(item["fallback"] for item in items),
        "manual_review": sum(item["manual_review"] for item in items),
        "priority_total": sum(item["priority"] for item in items),
        "priority_filled": sum(item["priority"] and item["fallback"] for item in items),
        "missing_problem_ids": [item["id"] for item in items if item["fallback"]],
        "items": items,
        "accepted_annotations": deepcopy(annotations),
    }


def coverage_for_review(review, findings):
    metadata = getattr(review, "coverage", {})
    # Recompute against current findings; stale metadata never supplies counts.
    return build_review_coverage(findings, metadata.get("accepted_annotations", {}))


def coverage_summary(coverage):
    return (
        f"检测问题总数：{coverage['total']}；AI 已解释数量：{coverage['ai_explained']}；"
        f"确定性补齐数量：{coverage['deterministic_filled']}；仍需验证或补齐解释数量：{coverage.get('verification_pending', coverage['manual_review'])}。\n\n"
        "统计口径：以上数字均按去重后的问题编号计算。AI 已解释与确定性补齐互斥且合计为总数；"
        "仍需验证或补齐解释是其中验证尚未确认或缺少完整 AI 解释的子集，可与前两项重叠；不等同于验证分类中的待人工复核。"
    )



def ensure_review_coverage(review, facts):
    """Legacy/no-AI text gets deterministic explanations, never invented AI credit."""
    findings = findings_from_facts(facts)
    if isinstance(review, ReviewText):
        current = coverage_for_review(review, findings)
        if current == review.coverage:
            return review
        entries = [{"finding_id": fid, **fields} for fid, fields in
                   current["accepted_annotations"].items()]
        return render_structured_review({"explanations": entries}, facts)
    coverage = build_review_coverage(findings)
    if not findings:
        return ReviewText(review or "本次没有正式 Finding。", coverage)
    by_id = {f["id"]: f for f in findings}
    lines = [str(review or "本次未生成 AI 解释。"), "### 覆盖检查与确定性补齐", coverage_summary(coverage),
             "正文未提供可校验的结构化解释记录，以下摘要由底层事实生成，不计入 AI 已解释。"]
    for problem in coverage["items"]:
        member = next((by_id[fid] for fid in problem["finding_ids"] if is_priority_finding(by_id[fid])), by_id[problem["finding_ids"][0]])
        explanation = deterministic_explanation(member)
        refs = problem["id"] + "（" + "、".join(problem["finding_ids"]) + "）"
        lines.append(f"- {refs}〔结构化事实补齐〕：原因：{explanation['cause']}影响：{explanation['potential_impact']}建议：{explanation['recommendation']}")
    return ReviewText("\n\n".join(lines), coverage)


def render_problem_index(coverage):
    lines = ["问题按 P- 编号去重统计；F- 编号保留每份底层证据，关联证据不重复计为独立问题。"]
    lines.extend("- " + item["id"] + " ← " + "、".join(item["finding_ids"]) for item in coverage["items"])
    return "\n".join(lines)


def legacy_data_issues(data_result):
    """Normalize legacy reports with the same rules already used by report.py."""
    issues = []

    for column, rate in data_result.get("missing_values", {}).items():
        level = "High" if rate >= 0.5 else "Medium" if rate >= 0.1 else "Low"
        issues.append({
            "level": level,
            "severity": level,
            "confidence": "High",
            "rule_id": "DQ_MISSING_VALUES",
            "category": "Missing Values",
            "type": "Missing Values",
            "column": column,
            "problem": f"{column} 存在缺失值",
            "detail": f"缺失率为 {rate:.2%}",
            "message": f"{column} 存在缺失值",
            "evidence": f"缺失率为 {rate:.2%}",
            "recommendation": "结合缺失机制选择处理策略。",
        })

    duplicate_rows = data_result.get("duplicate_rows", 0)
    if duplicate_rows:
        issues.append({
            "level": "Medium",
            "severity": "Medium",
            "confidence": "High",
            "rule_id": "DQ_DUPLICATE_ROWS",
            "category": "Duplicates",
            "type": "Potential Duplicate Records",
            "column": None,
            "problem": "存在重复样本",
            "detail": (
                f"发现 {duplicate_rows} 条重复记录，"
                f"占全部样本的 {data_result.get('duplicate_rate', 0):.2%}"
            ),
            "message": "存在内容完全相同的重复记录。",
            "evidence": f"发现 {duplicate_rows} 条重复记录。",
            "recommendation": "核对业务主键和重复来源；不要自动删除。",
        })

    return issues


def build_findings(code_result, data_results):
    """Normalize analyzer observations only. Confidence is never proof of runtime harm."""
    records = []
    for location in code_result or []:
        for issue in location.get("issues", []):
            records.append(("code", location.get("file"), location.get("cell"), issue))
    entries = data_results or []
    if isinstance(entries, dict):
        entries = [{"result": entries}]
    for entry in entries:
        result = entry.get("result", entry)
        formal = result.get("issues")
        if formal is None:
            formal = legacy_data_issues(result)
        for issue in list(formal) + list(entry.get("project_issues", [])):
            records.append(("data", entry.get("file"), None, issue))
    findings, occurrences = [], {}
    for origin, file, cell, issue in records:
        location = {"file": file, "cell": issue.get("cell", cell),
                    "line": issue.get("line"), "end_line": issue.get("end_line"),
                    "column": issue.get("column"),
                    "affected_locations": deepcopy(issue.get("affected_locations", []))}
        identity = json.dumps([origin, issue.get("rule_id"), issue.get("type"), location,
                               issue.get("source_snippet")], sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        occurrences[digest] = occurrences.get(digest, 0) + 1
        finding_id = "F-" + digest + (f"-{occurrences[digest]}" if occurrences[digest] > 1 else "")
        nature = issue.get("issue_nature", "")
        if origin == "code":
            status = "confirmed_error" if issue.get("type") == "Syntax Error" else "static_risk"
            if issue.get("action") == "Informational":
                status = "needs_review"
        else:
            status = ("confirmed_error" if nature in {"Confirmed Invalid", "Rule Violation"}
                      and issue.get("confidence") == "High" else "needs_review")
        # Preserve an explicit upstream verification state without reinterpretation.
        status = issue.get("verification_status") or status
        findings.append({
            "id": finding_id, "origin": origin, "rule_id": issue.get("rule_id"),
            "type": issue.get("type", "未分类问题"), "location": location,
            "evidence": deepcopy(issue.get("evidence") or issue.get("source_snippet") or issue.get("message", "")),
            "source_snippet": issue.get("source_snippet"),
            "severity": issue.get("severity", "Unspecified"),
            "confidence": issue.get("confidence", "Unspecified"),
            "recommendation": issue.get("recommendation") or "结合原始证据复核，并在测试环境验证。",
            "verification_status": status, "issue_nature": nature,
            "detected_count": issue.get("count"),
            "detected_rate": issue.get("rate"),
            "affected_rows": deepcopy(issue.get("affected_rows", [])),
            "message": issue.get("message", ""),
            "category": issue.get("category", ""),
            "action": issue.get("action"), "priority": issue.get("priority"),
            "deduplication_key": issue.get("deduplication_key"),
        })
    return sorted(findings, key=lambda item: item["id"])


def findings_from_facts(facts):
    if "findings" in facts:
        return deepcopy(facts["findings"])
    return build_findings(
        [{"file": item.get("file"), "cell": item.get("cell"), "issues": [item]}
         for item in facts.get("formal_code_issues", [])],
        [{"file": item.get("file"), "result": {"issues": item.get("formal_data_issues", [])}}
         for item in facts.get("datasets", [])],
    )


def render_finding_evidence(findings):
    """Authoritative fields are rendered by code, never copied from model output."""
    lines = []
    for item in findings:
        loc = item["location"]
        position = "；".join(f"{key}={value}" for key, value in loc.items() if value is not None and value != [])
        lines.extend([
            f"### {item['id']} · {item['type']}",
            f"- 位置：{position or '未提供位置'}",
            f"- 严重程度：{item['severity']}；置信度：{item['confidence']}；验证状态：{STATUS_LABELS.get(item['verification_status'], item['verification_status'])}",
            f"- 证据：{item['evidence'] or '未提供具体证据'}",
        ])
        if item.get("detected_count") is not None:
            lines.append(f"- 检测数量：{item['detected_count']}")
        if item.get("detected_rate") is not None:
            lines.append(f"- 检测比例（原始值）：{item['detected_rate']}")
        lines.append("")
    return "\n".join(lines)


# AI annotations cannot assert new facts. Quantities belong in canonical evidence.
ASSERTED_ERROR_RE = re.compile(
    r"已(?:经)?(?:确认|证实|确定)|已发生|必然|确定(?:性)?(?:错误|故障|(?:实体)?泄漏)|(?:导致|造成).{0,8}(?:错误|泄漏)|"
    r"(?:存在|发生|构成|属于).{0,6}(?:数据泄漏|错误|故障)|"
    r"\b(?:confirmed|proven|definite|definitely|certainly)\b|"
    r"(?:leakage|error|fault).{0,10}(?:occurs|occurred|confirmed|proven)", re.I,
)
ENTITY_RE = re.compile(r"实体泄漏|实体重复|entity (?:leakage|duplication|overlap)", re.I)
STRONG_IMPACT_RE = re.compile(
    r"(?:显著|严重|大幅|明显).{0,10}(?:下降|降低|提升|提高|高估|低估|偏差)|"
    r"(?:准确率|性能|accuracy|performance).{0,12}(?:显著|严重|大幅)|"
    r"\b(?:significantly|severely|substantially|dramatically)\b", re.I,
)


def sanitize_explanation(text, facts, finding=None):
    """Drop only unsupported clauses, with Chinese reasons; never rewrite evidence."""
    if not isinstance(text, str):
        return "", []
    safe, reasons = [], []
    allowed_text = json.dumps(facts, ensure_ascii=False, default=str)
    allowed_tokens = set(re.findall(r"[A-Za-z_]\w*", allowed_text))
    for clause in re.split(r"(?<=[。！？；;，,\n])|(?<=[.!?])\s+", text[:6000]):
        clause = clause.strip()
        if not clause:
            continue
        reason = None
        if re.search(
            r"(?<![A-Za-z_])\d+(?:\.\d+)?|[一二三四五六七八九十百千万]+(?:行|条|组|个|倍|百分点)", clause,
        ):
            reason = "数量和比例以对应 Finding 的原始证据为准，未采用模型补充的数值。"
        elif STRONG_IMPACT_RE.search(clause):
            reason = "未提供运行对照，无法确认模型性能变化的方向和幅度。"
        elif ASSERTED_ERROR_RE.search(clause):
            reason = "模型解释不能将检测风险升级为已证实错误，验证状态以原始 Finding 为准。"
        elif ENTITY_RE.search(clause) and not (finding and finding.get("issue_nature") == "Confirmed Entity Overlap"):
            reason = "相同车型或特征组合不能证明同一业务实体，实体重叠需经有效业务主键和划分验证。"
        elif UNSUPPORTED_BUSINESS_CONTEXT_RE.search(clause):
            reason = "项目未提供相应业务场景或损失依据，无法确认该推断。"
        elif re.search(r"严重程度|置信度|验证状态|检测数量|\bseverity\b|\bconfidence\b", clause, re.I):
            reason = "问题元数据以原始 Finding 为准，未采用模型重述。"
        elif any(token not in allowed_tokens for pair in re.findall(r"`([A-Za-z_]\w*)`|\b([A-Za-z_]\w*)(?=\.[A-Za-z_]|\[|\()", clause) for token in pair if token):
            reason = "输入中未提供该变量，无法确认其作用。"
        elif any(name not in allowed_text for name in re.findall(r"[\w./-]+\.(?:py|ipynb|csv)\b", clause)):
            reason = "输入中未提供该文件，无法确认其位置。"
        if reason:
            reasons.append(reason)
        else:
            safe.append(clause)
    return re.sub(r"([，。！？；]) +", r"\1", " ".join(safe)), list(dict.fromkeys(reasons))


def render_structured_review(payload, facts):
    """Join model explanations by ID; omitted/unknown IDs never alter formal findings."""
    findings = findings_from_facts(facts)
    known = {item["id"]: item for item in findings}
    annotations, limitations = {}, []
    if not isinstance(payload, dict):
        payload = {}
        limitations.append("模型未返回有效的结构化解释；以下保留全部原始诊断和建议。")
    entries = payload.get("explanations", [])
    if not isinstance(entries, list):
        entries = []
    for entry in entries[:len(findings) + 20]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("finding_id"), str)
                or entry["finding_id"] not in known):
            limitations.append("已忽略未对应原始 Finding ID 的模型结论。")
            continue
        fid = entry["finding_id"]
        if fid in annotations:
            limitations.append("同一 Finding 的重复解释未重复采用。")
            continue
        annotations[fid] = {}
        for key in ("cause", "potential_impact", "recommendation"):
            clean, notes = sanitize_explanation(entry.get(key, ""), facts, known[fid])
            clean, consistency_notes = validate_agent_consistency(clean, facts, ensure_findings=False)
            annotations[fid][key] = clean
            limitations.extend(notes + consistency_notes)
    coverage = build_review_coverage(findings, annotations)
    lines = ["## 1. 项目总结", "本报告以静态分析事实为依据；AI 仅补充解释，未执行模型训练或性能对照。",
             coverage_summary(coverage), "", "## 2. 发现的问题", render_problem_index(coverage),
             render_finding_evidence(findings) or "本次未产生结构化 Finding；这不代表项目没有问题。"]
    for heading, key in [("## 3. 问题原因", "cause"), ("## 4. 影响分析", "potential_impact"), ("## 5. 优化建议", "recommendation")]:
        lines.extend(["", heading])
        grouped = {}
        for problem in coverage["items"]:
            members = sorted([known[fid] for fid in problem["finding_ids"]], key=lambda f: (not is_priority_finding(f), f["id"]))
            source = next((annotations[fid][key] for fid in problem["finding_ids"]
                           if annotations.get(fid, {}).get(key)), None)
            value = source or deterministic_explanation(members[0])[key]
            # Important problems always have individual explanations. Other
            # problems may share a concise category/interpretation summary.
            group_key = (problem["id"] if problem["priority"] else problem["category"], value,
                         "AI 解释" if source else "结构化事实补齐")
            grouped.setdefault(group_key, []).append(problem)
        for (_, value, origin), problems in grouped.items():
            refs = "、".join(p["id"] + "（" + "、".join(p["finding_ids"]) + "）" for p in problems)
            field_label = {"cause": "原因", "potential_impact": "潜在影响", "recommendation": "建议"}[key]
            lines.append(f"- {refs}〔{origin}〕：{field_label}：{value}")
    suggestions = payload.get("unverified_suggestions", [])
    if isinstance(suggestions, list):
        for suggestion in suggestions[:10]:
            if not isinstance(suggestion, str):
                continue
            clean, notes = sanitize_explanation(suggestion, facts)
            clean, consistency_notes = validate_agent_consistency(clean, facts, ensure_findings=False)
            limitations.extend(notes + consistency_notes)
            if clean:
                lines.append("- 待验证建议（不计入原始检测数量）：" + clean)
    lines.extend(["", "## 6. 修改代码示例", "以 Repair 模块中经校验的修改建议为准；AI Review 不生成未经验证的可执行 Patch。",
                  "", "### 验证限制", "未执行运行对照，无法确认模型性能影响；数据或变量未提供时不能推断其取值和业务含义。"])
    if coverage["missing_problem_ids"]:
        limitations.append("以下问题未获得完整的经校验 AI 解释，已用结构化事实补齐：" +
                           "、".join(coverage["missing_problem_ids"]))
    for problem in coverage["items"]:
        if problem["missing_finding_ids"]:
            limitations.append(problem["id"] + " 未被模型提及的 Finding：" + "、".join(problem["missing_finding_ids"]))
    lines.extend("- " + note for note in dict.fromkeys(limitations))
    return ReviewText("\n\n".join(lines), coverage)



def build_analysis_facts(
    code_result,
    data_results,
    data_flow,
    quantified_impacts=None,
    duplicate_split_analysis=None,
    business_rule_results=None,
    artifact_analysis=None,
    skipped_files=None,
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
            "column_names": list(result.get("data_types", {})),
            "missing_values": result.get("missing_values", {}),
            "duplicate_rows": result.get("duplicate_rows", 0),
            "duplicate_rate": result.get("duplicate_rate", 0),
            "duplicate_groups": result.get("duplicate_groups", 0),
            "formal_data_issues": result.get("issues", []),
        })
    return {
        "findings": build_findings(code_result, data_results),
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
        "analysis_scope": {
            "skipped_files": list(skipped_files or []),
            "coverage_complete": not bool(skipped_files),
            "statement": (
                "All discovered supported files were analyzed."
                if not skipped_files else
                "Project coverage is partial; skipped dependencies/files are listed with reasons."
            ),
        },
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
    duplicate = facts.get("duplicate_split_analysis", {})
    for scope in [duplicate, *(duplicate.get("scopes") or {}).values()]:
        for key in ("duplicate_groups", "cross_split_duplicate_groups", "shared_records"):
            if scope.get(key) is not None:
                values.add(str(scope[key]))
    modeling = duplicate.get("modeling_data", {})
    for key in ("raw_rows", "cleaned_rows", "train_rows", "test_rows"):
        if modeling.get(key) is not None:
            values.add(str(modeling[key]))
    roles = facts.get("split_role_analysis", {})
    if roles.get("estimated_test_rows") is not None:
        values.add(str(roles["estimated_test_rows"]))
    if roles.get("test_ratio") is not None:
        percent = float(roles["test_ratio"]) * 100
        values.update({f"{percent:.1f}", f"{percent:.2f}"})
    return values


def validate_agent_consistency(agent_text, facts, *, ensure_findings=True):
    """Correct individual clauses, retaining neighboring evidence and useful advice."""
    facts = facts or {}
    if ensure_findings:
        agent_text = ensure_review_coverage(agent_text, facts)
    if not agent_text:
        return agent_text, []
    corrections, output = [], []
    metrics = [str(item) for item in facts.get("detected_metrics", []) if item]
    trusted = set(render_finding_evidence(findings_from_facts(facts)).splitlines())
    allowed = _allowed_quantities(facts)
    typed_coverage = coverage_for_review(agent_text, findings_from_facts(facts)) if isinstance(agent_text, ReviewText) else None
    if typed_coverage:
        trusted.update(coverage_summary(typed_coverage).splitlines())
        # Structured model explanations were already sanitized before rendering;
        # these local lines also contain authoritative rule names and fallback text.
        trusted.update(line for line in str(agent_text).splitlines()
                       if line.startswith("- P-") and ("〔AI 解释〕" in line or "〔结构化事实补齐〕" in line))
    for line in str(agent_text).splitlines():
        if line in trusted or line.startswith("##") or line == f"原始 Finding 数量：{len(findings_from_facts(facts))}。":
            output.append(line)
            continue
        pieces = re.split(r"(?<=[。！？；;，,])|(?<=[.!?])\s+", line)
        kept = []
        for piece in pieces:
            corrected = piece
            fairness_contradiction = any(p.search(piece) for p in FAIRNESS_CONTRADICTIONS)
            if facts.get("detected_split") and any(p.search(piece) for p in SPLIT_CONTRADICTIONS):
                corrected = "已检测到训练/测试划分；覆盖范围以静态数据流事实为准。"
            elif facts.get("fairness", {}).get("evaluation_exists") and fairness_contradiction:
                corrected = "已检测到公平性评估；具体指标覆盖仍需结合对象类型复核。"
            elif (facts.get("evaluation_detected") or metrics) and not fairness_contradiction and any(p.search(piece) for p in EVALUATION_CONTRADICTIONS):
                corrected = "已检测到评价流程（" + ("、".join(metrics) or "predict/score") + "）；覆盖是否充分需单独复核。"
            elif STRONG_IMPACT_RE.search(piece) or UNSUPPORTED_IMPACT_RE.search(piece):
                corrected = "当前证据不能确认模型性能变化的方向和幅度，需通过运行对照验证。"
            elif UNSUPPORTED_BUSINESS_CONTEXT_RE.search(piece):
                corrected = "项目未记录相应业务场景或代价，无法确认该推断。"
            elif ENTITY_RE.search(piece) and ASSERTED_ERROR_RE.search(piece):
                corrected = "相同车型或特征组合仅表明模式复用；是否属于同一实体需核对有效业务主键与划分。"
            elif any(m.group("number") not in allowed for m in QUANTIFIED_CLAIM_RE.finditer(piece)):
                corrected = "该项数值缺少对应事实依据，未纳入结论；请以原始检测证据为准。"
            elif ASSERTED_ERROR_RE.search(piece) and any(f["verification_status"] != "confirmed_error" for f in findings_from_facts(facts)):
                corrected = "该项仅由静态证据支持，尚不能确认运行时错误；需人工复核。"
            if corrected != piece:
                corrections.append("已精确修正局部不可信结论：" + corrected)
            kept.append(corrected)
        output.append("".join(kept).replace("Unable to verify automatically", "现有输入不足，无法自动确认").replace("API Unverified", "接口调用尚未验证"))
    result = "\n".join(output)
    # Legacy Markdown callers also retain full canonical coverage.
    findings = findings_from_facts(facts)
    missing = [f for f in findings if render_finding_evidence([f]).strip() not in result]
    if ensure_findings and missing:
        result += "\n\n### 原始 Finding 补充（保持底层检测结果）\n" + render_finding_evidence(missing)
        corrections.append("已补回 AI 正文遗漏的原始 Finding。")
    if isinstance(agent_text, ReviewText):
        result = ReviewText(result, typed_coverage)
    return result, list(dict.fromkeys(corrections))



FINAL_REPORT_SCHEMA = "final-report-v1"


def report_fingerprint(report):
    content = {key: value for key, value in report.items() if key != "final_fingerprint"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def is_final_report(report):
    return bool(report and report.get("schema_version") == FINAL_REPORT_SCHEMA
                and report.get("final_fingerprint") == report_fingerprint(report))


def _issue_projection(finding, problem_id, original=None):
    issue = deepcopy(original or {})
    loc = finding["location"]
    issue.update({
        "id": finding["id"], "finding_id": finding["id"], "problem_id": problem_id,
        "rule_id": finding["rule_id"], "type": finding["type"],
        "file": loc["file"], "cell": loc["cell"], "line": loc["line"], "end_line": loc["end_line"],
        "column": loc["column"], "severity": finding["severity"], "confidence": finding["confidence"],
        "evidence": finding["evidence"], "source_snippet": finding["source_snippet"],
        "message": finding["message"], "reason": finding["message"],
        "recommendation": finding["recommendation"], "verification_status": finding["verification_status"],
        "verification_label": STATUS_LABELS.get(finding["verification_status"], "需要人工复核"),
        "category": finding.get("category") or issue.get("category", "Quality"),
        "action": finding.get("action") or "Review",
        "count": finding.get("detected_count"), "rate": finding.get("detected_rate"),
        "affected_rows": finding.get("affected_rows", []),
        "problem": finding["type"], "detail": finding["evidence"],
        "affected_files": issue.get("affected_files") or [loc["file"]],
    })
    return issue


def finalize_report_projection(checked):
    """Project every displayed issue/number from final findings and analyzer facts."""
    findings, facts = checked["findings"], checked["analysis_facts"]
    problems = group_findings(findings)
    membership = {fid: problem["id"] for problem in problems for fid in problem["finding_ids"]}
    source = checked.get("source_snapshot", {})
    if "overview" in source:
        checked["project_overview"] = deepcopy(source["overview"])
    if "health" in source:
        checked["overall"] = deepcopy(source["health"])
    raw_code = source.get("code_result", [])
    raw_data = deepcopy(source.get("data_results", []))
    if not raw_data:
        raw_data = [{"file": d["file"], "result": deepcopy(d)} for d in checked.get("data_quality", {}).get("datasets", [])]
    rows, code = [], []
    data_by_file = {}
    project_issues = []
    for finding in findings:
        loc = finding["location"]
        original = next((issue for location in raw_code if location.get("file") == loc["file"] and location.get("cell") == loc["cell"]
                         for issue in location.get("issues", []) if issue.get("rule_id") == finding["rule_id"]
                         and issue.get("line") == loc["line"]), {}) if finding["origin"] == "code" else {}
        if finding["origin"] == "data":
            original = next((issue for entry in raw_data if entry.get("file") == loc["file"]
                             for issue in entry.get("result", entry).get("issues", [])
                             if issue.get("rule_id") == finding["rule_id"] and issue.get("column") == loc["column"]), {})
        issue = _issue_projection(finding, membership[finding["id"]], original)
        rows.append(issue)
        if finding["origin"] == "code":
            code.append({"file": loc["file"], **({"cell": loc["cell"]} if loc["cell"] is not None else {}), "issues": [issue]})
        elif finding.get("category") == "Multi-file Consistency":
            project_issues.append(issue)
        else:
            data_by_file.setdefault(loc["file"], []).append(issue)
    profiles = {d.get("file"): d for d in facts.get("datasets", [])}
    for entry in raw_data:
        result = entry.get("result", entry)
        verified = profiles.get(entry.get("file"), {})
        for field in ("missing_values", "duplicate_rows", "duplicate_rate", "duplicate_groups"):
            if field in verified:
                result[field] = deepcopy(verified[field])
        shape = dict(result.get("shape", {}))
        for key in ("rows", "columns"):
            if key in verified:
                shape[key] = verified[key]
        result["shape"] = shape
        result["issues"] = data_by_file.get(entry.get("file"), [])
        entry["project_issues"] = project_issues
    code_flat = [issue for location in code for issue in location["issues"]]
    checked.setdefault("code_quality", {})["issues"] = code_flat
    checked["code_quality"]["issue_count"] = sum(i.get("score_impact") != 0 and i.get("action") != "Informational" for i in code_flat)
    checked["code_quality"]["informational_count"] = len(code_flat) - checked["code_quality"]["issue_count"]
    for dataset in checked.setdefault("data_quality", {}).setdefault("datasets", []):
        normalized = next((e.get("result", e) for e in raw_data if e.get("file") == dataset.get("file")), {})
        dataset.update(deepcopy(normalized))
        from report import _data_quality_score
        dataset["quality_score"] = _data_quality_score(normalized)
    checked["data_quality"]["project_issues"] = project_issues
    points = checked.get("overall", {}).get("breakdown", {}).get("代码质量", 0)
    checked["code_quality"]["score"] = round(points / 35 * 100) if checked.get("project_overview", {}).get("python_file_count") else 0
    counts = {"total": len(problems), "confirmed_error": 0, "static_risk": 0, "needs_review": 0}
    by_id = {f["id"]: f for f in findings}
    for problem in problems:
        states = {by_id[fid]["verification_status"] for fid in problem["finding_ids"]}
        state = "needs_review" if "needs_review" in states or not states <= set(STATUS_LABELS) else "static_risk" if "static_risk" in states else "confirmed_error"
        problem["verification_status"] = state
        counts[state] += 1
    repairs = checked.get("repair_suggestions", [])
    counts.update({"ai_explained": checked["review_coverage"]["ai_explained"],
                   "verification_pending": checked["review_coverage"]["manual_review"],
                   "repairs": len(repairs), "validated_repairs": sum(r.get("validation_status") in {"Statically Validated", "Runtime Verified"} for r in repairs)})
    checked["conclusions"] = {"counting_unit": "deduplicated_problem_id", "counts": counts,
                              "problems": problems, "finding_rows": rows,
                              "patch_available": bool(checked.get("patch_text")),
                              "patch_status": "已生成经过验证的补丁" if checked.get("patch_text") else "未生成可导出的补丁"}
    checked["view"] = {
        "code_result": code, "data_results": raw_data, "data_result": raw_data[0]["result"] if raw_data else None,
        "data_consistency_issues": project_issues, "analysis_facts": facts,
        "data_flow": facts.get("data_flow", {}), "quantified_impacts": facts.get("quantified_impacts", []),
        "artifact_analysis": facts.get("artifact_analysis", {}), "health": checked.get("overall", {}),
        "project_overview": checked.get("project_overview", {}), "repair_suggestions": repairs,
        "patch_text": checked["patch_text"], "agent_report": checked["ai_review"],
        "enhancement_status": checked.get("enhancement_status", {}),
    }
    checked["schema_version"] = FINAL_REPORT_SCHEMA
    checked["final_fingerprint"] = report_fingerprint(checked)
    return checked


def conclusion_summary_lines(report):
    c = report["conclusions"]["counts"]
    return [f"检测问题总数：{c['total']}；已证实错误：{c['confirmed_error']}；风险提示：{c['static_risk']}；待人工复核：{c['needs_review']}。",
            "统计按去重后的 P- 问题编号计算，三类验证结论互斥且合计为总数；原始 F- 证据逐条保留。",
            f"修复建议：{c['repairs']}；静态或运行验证通过：{c['validated_repairs']}；{report['conclusions']['patch_status']}。"]


def finding_summary_line(issue):
    loc = " · ".join(str(value) for value in [issue.get("file"),
        f"Cell {issue['cell']}" if issue.get("cell") is not None else None,
        f"Line {issue['line']}" if issue.get("line") is not None else None,
        f"Column {issue['column']}" if issue.get("column") is not None else None] if value is not None)
    return (f"{issue['id']} / {issue['problem_id']} · {issue['type']} · {loc} · "
            f"严重程度：{issue['severity']} · 置信度：{issue['confidence']} · {issue['verification_label']} · 证据：{issue['evidence']}")



def validate_report_consistency(report):
    """Return a corrected report model before Markdown/PDF renderers consume it."""
    if is_final_report(report):
        return deepcopy(report)
    checked = deepcopy(report)
    facts = deepcopy(checked.get("analysis_facts", {}))
    blocked = set()
    if facts.get("detected_split"):
        blocked.add("CODE_NO_VALIDATION_SPLIT")
    if facts.get("evaluation_detected"):
        blocked.add("CODE_NO_EVALUATION")
    facts["findings"] = [f for f in findings_from_facts(facts) if f.get("rule_id") not in blocked]
    facts["formal_code_issues"] = [i for i in facts.get("formal_code_issues", []) if i.get("rule_id") not in blocked]
    checked["analysis_facts"] = facts
    review = checked.get("ai_review", "")
    recorded = checked.get("review_coverage", {})
    if not isinstance(review, ReviewText) and recorded.get("accepted_annotations"):
        entries = [{"finding_id": fid, **fields} for fid, fields in recorded["accepted_annotations"].items()]
        review = render_structured_review({"explanations": entries}, facts)
    if checked.get("source_snapshot", {}).get("structured_review") is False:
        review = render_structured_review({"explanations": []}, facts)
        review = ReviewText("本次未生成可校验的结构化 AI 解释；以下保留完整静态诊断与确定性摘要。\n\n" + review, review.coverage)
    corrected_ai, corrections = validate_agent_consistency(review, facts)
    checked["ai_review"] = corrected_ai
    checked["findings"] = findings_from_facts(facts)
    checked["review_coverage"] = coverage_for_review(corrected_ai, checked["findings"])

    # Revalidate executable suggestions at the final report boundary.  This
    # prevents a later formatter/caller from reintroducing a field or variable
    # that was not available at the proposed execution location.
    if checked.get("repair_suggestions"):
        from repair_validator import validate_repair_suggestions

        pseudo_results = [
            {"file": dataset.get("file"), "result": {
                "data_types": {column: "unknown" for column in dataset.get("column_names", [])}
            }}
            for dataset in facts.get("datasets", [])
        ]
        checked["repair_suggestions"] = validate_repair_suggestions(
            checked["repair_suggestions"], facts.get("data_flow", {}), pseudo_results,
        )

    # Cached or caller-supplied diffs must not bypass current repair validation.
    from patch_generator import build_unified_patch
    checked["patch_text"] = build_unified_patch(checked.get("repair_suggestions", []))
    checked["repair_validation_policy"] = "exact-source-p1"

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
    visible_issues = checked.get("code_quality", {}).get("issues", [])
    checked["code_quality"]["issue_count"] = sum(
        issue.get("score_impact") != 0 and issue.get("action") != "Informational"
        for issue in visible_issues
    )
    checked["code_quality"]["informational_count"] = sum(
        issue.get("score_impact") == 0 or issue.get("action") == "Informational"
        for issue in visible_issues
    )
    checked["consistency_corrections"] = list(dict.fromkeys(corrections))
    return finalize_report_projection(checked)
