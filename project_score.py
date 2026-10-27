"""Explainable severity-aware project health scoring."""

from __future__ import annotations

from collections import Counter, defaultdict


CODE_WEIGHTS = {"High": 10.0, "Medium": 4.0, "Low": 1.25}
DATA_WEIGHTS = {"High": 8.0, "Medium": 3.5, "Low": 0.75}


def _legacy_code_severity(issue):
    return {
        "Syntax Error": "High",
        "Data Leakage Risk": "High",
        "Potential Data Leakage": "Medium",
        "Missing Value Handling Risk": "Medium",
    }.get(issue.get("type"), "Low")


def _bounded_penalty_result(issues, weights, category_cap, total_cap, fallback_severity=None):
    """Return a capped penalty plus auditable, non-informational sources."""
    by_rule = defaultdict(list)
    metadata = {}
    for issue in issues:
        if issue.get("score_impact") == 0 or issue.get("action") == "Informational":
            continue
        severity = issue.get("severity") or (
            fallback_severity(issue) if fallback_severity else "Low"
        )
        category = issue.get("category", "General")
        rule_id = (issue.get("root_cause_id") or issue.get("deduplication_key")
                   or issue.get("rule_id") or issue.get("type", "UNKNOWN"))
        by_rule[(category, rule_id)].append(weights.get(severity, weights["Low"]))
        item = metadata.setdefault((category, rule_id), {
            "category": category,
            "root_cause": rule_id,
            "rule_ids": [],
            "types": [],
            "severities": [],
        })
        for key, value in (("rule_ids", issue.get("rule_id")),
                           ("types", issue.get("type")),
                           ("severities", severity)):
            if value and value not in item[key]:
                item[key].append(value)

    by_category = defaultdict(float)
    details = []
    for (category, _), values in by_rule.items():
        # First occurrence carries the real weight; repeats add only a small bounded signal.
        rule_penalty = max(values) + min(2.0, max(0, len(values) - 1) * 0.5)
        by_category[category] += rule_penalty
        details.append({
            **metadata[(category, _)],
            "occurrences": len(values),
            "raw_penalty": rule_penalty,
        })
    applied = min(total_cap, sum(min(category_cap, value) for value in by_category.values()))
    return applied, sorted(details, key=lambda item: (-item["raw_penalty"], item["category"], str(item["root_cause"])))


def _bounded_penalty(issues, weights, category_cap, total_cap, fallback_severity=None):
    """Cap repeated rule/category penalties so one root cause cannot zero a score."""
    return _bounded_penalty_result(
        issues, weights, category_cap, total_cap, fallback_severity,
    )[0]


def _severity_counts(issues, fallback=None):
    counts = Counter()
    for issue in issues:
        if issue.get("score_impact") == 0 or issue.get("action") == "Informational":
            continue
        severity = issue.get("severity") or (fallback(issue) if fallback else "Low")
        counts[severity] += 1
    return counts


def calculate_project_score(code_result, data_results, code_files_count=0, dataset_graph=None):
    """Score code (35), data (35), and observable project completeness (30)."""
    if isinstance(data_results, dict):
        data_results = [data_results]
    entries = list(data_results or [])
    reports = [entry.get("result", entry) for entry in entries]
    has_code = code_files_count > 0

    code_issues = [
        issue
        for location in code_result or []
        for issue in location.get("issues", [])
    ]
    code_penalty, code_penalty_details = _bounded_penalty_result(
        code_issues, CODE_WEIGHTS, category_cap=14.0, total_cap=35.0,
        fallback_severity=_legacy_code_severity,
    )
    code = max(0, round(35 - code_penalty)) if has_code else 0

    data_scores = []
    data_penalty_details = []
    all_data_issues = []
    for entry, report in zip(entries, reports):
        issues = list(report.get("issues", []))
        if not issues:
            # Compatibility for legacy result dictionaries.
            for rate in report.get("missing_values", {}).values():
                issues.append({
                    "rule_id": "DQ_MISSING_VALUES", "category": "Missing Values",
                    "severity": "High" if rate >= 0.5 else "Medium" if rate >= 0.1 else "Low",
                })
            if report.get("duplicate_rows", 0):
                rate = float(report.get("duplicate_rate", 0))
                issues.append({
                    "rule_id": "DQ_DUPLICATE_ROWS", "category": "Duplicates",
                    "severity": "High" if rate > 0.10 else "Medium" if rate >= 0.01 else "Low",
                })
            if report.get("shape", {}).get("rows", 0) == 0:
                issues.append({"rule_id": "DQ_EMPTY_DATASET", "category": "Structure", "severity": "High"})
        penalty, details = _bounded_penalty_result(
            issues, DATA_WEIGHTS, category_cap=10.0, total_cap=35.0,
        )
        data_scores.append(max(0, 35 - penalty))
        data_penalty_details.append({
            "file": entry.get("file"), "applied_penalty": penalty, "sources": details,
        })
        all_data_issues.extend(issues)

    project_issues = []
    for entry in entries:
        project_issues.extend(entry.get("project_issues", []))
    project_penalty = 0
    project_penalty_details = []
    if data_scores and project_issues:
        project_penalty, project_penalty_details = _bounded_penalty_result(
            project_issues, DATA_WEIGHTS, category_cap=4.0, total_cap=5.0,
        )
        data_scores[0] = max(0, data_scores[0] - project_penalty)
        all_data_issues.extend(project_issues)
    groups = {entry.get('file', str(i)): entry.get('file', str(i)) for i, entry in enumerate(entries)}
    def root(name):
        while groups[name] != name:
            name = groups[name]
        return name
    for dataset in (dataset_graph or {}).get('datasets', []):
        child = dataset['file']
        for parent in dataset.get('parents', []):
            if child in groups and parent in groups:
                groups[root(child)] = root(parent)
    group_scores = defaultdict(list)
    for i, (entry, value) in enumerate(zip(entries, data_scores)):
        group_scores[root(entry.get('file', str(i)))].append(value)
    effective_scores = [min(values) for values in group_scores.values()]
    data = round(sum(effective_scores) / len(effective_scores)) if effective_scores else 0

    completeness = (15 if has_code else 0) + (15 if reports else 0)
    breakdown = {"代码质量": code, "数据质量": data, "项目完整性": completeness}
    code_counts = _severity_counts(code_issues, _legacy_code_severity)
    data_counts = _severity_counts(all_data_issues)
    explanations = [
        (
            f"代码质量 {code}/35：按严重程度、rule_id 与类别上限计算；"
            f"High {code_counts['High']}、Medium {code_counts['Medium']}、Low {code_counts['Low']}。"
        ) if has_code else "代码质量 0/35：未发现可分析的代码文件。",
        (
            f"数据质量 {data}/35：基于 {len(reports)} 个 CSV 的缺失、重复、非有限值、"
            f"极端值、类型、特征与相关性；High {data_counts['High']}、"
            f"Medium {data_counts['Medium']}、Low {data_counts['Low']}。"
        ) if reports else "数据质量 0/35：未发现可分析的 CSV。",
        f"项目完整性 {completeness}/30：可分析代码文件 {'已提供' if has_code else '未提供'}，"
        f"CSV 数据文件 {'已提供' if reports else '未提供'}。",
    ]
    if dataset_graph and len(reports) > 1:
        explanations.append(f"同来源读写血缘共 {len(group_scores)} 组：组内取最低数据质量分，再对各组等权平均；不按派生文件数或重复行数扩大样本权重。")
    if code_penalty_details:
        explanations.append(
            "代码扣分来源（扣分上限前）：" + "；".join(
                f"{item['category']}/{','.join(item['rule_ids']) or item['root_cause']} "
                f"{item['raw_penalty']:g}分（{item['occurrences']}处）"
                for item in code_penalty_details
            ) + f"；实际应用 {code_penalty:g}分。"
        )
    return {
        "score": sum(breakdown.values()),
        "breakdown": breakdown,
        "explanation": explanations,
        "caveat": "启发式静态评分；仅反映已导入文件的检查结果，不代表项目真实模型性能或运行表现。",
        "dataset_scoring": {"unit": "observed_lineage_group", "groups": dict(groups),
                            "group_count": len(group_scores), "within_group": "minimum", "across_groups": "equal_weight_mean"},
        "penalty_details": {
            "code": {"applied_penalty": code_penalty, "sources": code_penalty_details},
            "data": data_penalty_details,
            "project": {"applied_penalty": project_penalty, "sources": project_penalty_details},
            "policy": {
                "informational_excluded": True,
                "repeated_root_cause_increment_cap": 2.0,
                "code_category_cap": 14.0,
                "data_category_cap": 10.0,
            },
        },
    }
