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


def _bounded_penalty(issues, weights, category_cap, total_cap, fallback_severity=None):
    """Cap repeated rule/category penalties so one root cause cannot zero a score."""
    by_rule = defaultdict(list)
    for issue in issues:
        if issue.get("score_impact") == 0 or issue.get("action") == "Informational":
            continue
        severity = issue.get("severity") or (
            fallback_severity(issue) if fallback_severity else "Low"
        )
        category = issue.get("category", "General")
        rule_id = issue.get("root_cause_id") or issue.get("rule_id") or issue.get("type", "UNKNOWN")
        by_rule[(category, rule_id)].append(weights.get(severity, weights["Low"]))

    by_category = defaultdict(float)
    for (category, _), values in by_rule.items():
        # First occurrence carries the real weight; repeats add only a small bounded signal.
        rule_penalty = max(values) + min(2.0, max(0, len(values) - 1) * 0.5)
        by_category[category] += rule_penalty
    return min(total_cap, sum(min(category_cap, value) for value in by_category.values()))


def _severity_counts(issues, fallback=None):
    counts = Counter()
    for issue in issues:
        if issue.get("score_impact") == 0 or issue.get("action") == "Informational":
            continue
        severity = issue.get("severity") or (fallback(issue) if fallback else "Low")
        counts[severity] += 1
    return counts


def calculate_project_score(code_result, data_results, code_files_count=0):
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
    code_penalty = _bounded_penalty(
        code_issues, CODE_WEIGHTS, category_cap=14.0, total_cap=35.0,
        fallback_severity=_legacy_code_severity,
    )
    code = max(0, round(35 - code_penalty)) if has_code else 0

    data_scores = []
    all_data_issues = []
    for report in reports:
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
        penalty = _bounded_penalty(
            issues, DATA_WEIGHTS, category_cap=10.0, total_cap=35.0,
        )
        data_scores.append(max(0, 35 - penalty))
        all_data_issues.extend(issues)

    project_issues = []
    for entry in entries:
        project_issues.extend(entry.get("project_issues", []))
    if data_scores and project_issues:
        project_penalty = _bounded_penalty(
            project_issues, DATA_WEIGHTS, category_cap=4.0, total_cap=5.0,
        )
        data_scores[0] = max(0, data_scores[0] - project_penalty)
        all_data_issues.extend(project_issues)
    data = round(sum(data_scores) / len(data_scores)) if data_scores else 0

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
    return {
        "score": sum(breakdown.values()),
        "breakdown": breakdown,
        "explanation": explanations,
        "caveat": "启发式静态评分；仅反映已导入文件的检查结果，不代表项目真实模型性能或运行表现。",
    }
