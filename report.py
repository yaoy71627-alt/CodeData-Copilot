"""Canonical analysis report data plus Markdown and PDF renderers."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import re
from xml.sax.saxutils import escape

from data_flow_analyzer import flow_summary_text


def count_code_issues(code_result):
    return sum(len(item.get("issues", [])) for item in code_result or [])


def build_data_issues(data_result):
    if not data_result:
        return []

    canonical = data_result.get("issues")
    if canonical is not None:
        return [
            {
                **issue,
                "level": issue.get("severity", "Low"),
                "problem": issue.get("type", "Data quality issue"),
                "detail": issue.get("evidence") or issue.get("message", ""),
            }
            for issue in canonical
        ]

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


def summarize_results(code_result, data_result):
    return {
        "code_issue_count": count_code_issues(code_result),
        "data_issue_count": len(build_data_issues(data_result)),
        "rows": data_result.get("shape", {}).get("rows", 0) if data_result else 0,
        "columns": data_result.get("shape", {}).get("columns", 0) if data_result else 0,
    }


def _severity(issue_type):
    return {
        "Syntax Error": "High",
        "Data Leakage Risk": "High",
        "Potential Data Leakage": "Medium",
        "Missing Value Handling Risk": "Medium",
    }.get(issue_type, "Low")


def _data_quality_score(result):
    shape = result.get("shape", {})
    if not shape.get("rows") or not shape.get("columns"):
        return 0
    weights = {"High": 18, "Medium": 7, "Low": 2}
    grouped = {}
    for issue in build_data_issues(result):
        key = (issue.get("category"), issue.get("rule_id", issue.get("type")))
        grouped.setdefault(key, []).append(weights.get(issue.get("severity", "Low"), 2))
    penalty = sum(max(values) + min(3, len(values) - 1) for values in grouped.values())
    return max(0, round(100 - min(100, penalty)))


def build_report_data(
    project_overview,
    code_result,
    data_results,
    agent_report,
    repair_suggestions,
    patch_text,
    health,
    analysis_facts=None,
):
    """Build the single data model consumed by every report format."""
    overview = dict(project_overview or {})
    repairs = list(repair_suggestions or [])
    code_issues = []
    for location in code_result or []:
        file_name = location.get("file", "Python file")
        notebook_cell = location.get("cell")
        for issue in location.get("issues", []):
            issue_type = issue.get("type", "Code issue")
            repair = next(
                (
                    item for item in repairs
                    if item.get("problem") == issue_type
                    and str(item.get("location", "")).startswith(file_name)
                ),
                None,
            )
            code_issues.append({
                "rule_id": issue.get("rule_id"),
                "category": issue.get("category", "Code Quality"),
                "type": issue_type,
                "severity": issue.get("severity", _severity(issue_type)),
                "confidence": issue.get("confidence", "Medium"),
                "file": file_name,
                "affected_files": issue.get("affected_files") or location.get("affected_files") or [file_name],
                "affected_locations": issue.get("affected_locations", []),
                "line": issue.get("line"),
                "end_line": issue.get("end_line"),
                "cell": notebook_cell,
                "reason": issue.get("message", "静态检查发现需要人工复核的代码风险。"),
                "evidence": issue.get("evidence") or issue.get("source_snippet"),
                "source_snippet": issue.get("source_snippet"),
                "recommendation": (
                    issue.get("recommendation") or (repair.get("explanation") if repair
                    else "结合项目上下文复核，并在测试环境验证修改。"
                    )
                ),
            })

    datasets = []
    for entry in data_results or []:
        result = entry.get("result", entry)
        high_correlations = [
            {
                "features": f"{item.get('feature_a')} / {item.get('feature_b')}",
                "value": float(item.get("correlation", 0)),
            }
            for item in result.get("high_correlation_pairs", [])
        ]
        if not high_correlations:
            correlation = result.get("correlation", {})
            seen = set()
            for left, values in correlation.items():
                for right, value in values.items():
                    pair = tuple(sorted((str(left), str(right))))
                    if left != right and pair not in seen and abs(float(value)) >= 0.95:
                        seen.add(pair)
                        high_correlations.append({"features": f"{left} / {right}", "value": float(value)})
        datasets.append({
            "file": entry.get("file", "CSV"),
            "shape": result.get("shape", {}),
            "missing_values": result.get("missing_values", {}),
            "duplicate_rows": result.get("duplicate_rows", 0),
            "duplicate_rate": result.get("duplicate_rate", 0),
            "duplicate_groups": result.get("duplicate_groups", 0),
            "duplicate_split_analysis": result.get("duplicate_split_analysis", {}),
            "data_types": result.get("data_types", {}),
            "high_correlations": high_correlations,
            "issues": build_data_issues(result),
            "numeric_statistics": result.get("numeric_statistics", {}),
            "quality_score": _data_quality_score(result),
        })

    health = dict(health or {})
    code_points = health.get("breakdown", {}).get("代码质量", 0)
    code_score = round(code_points / 35 * 100) if overview.get("python_file_count") else 0
    return {
        "title": "CodeData-Copilot Analysis Report",
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "project_overview": overview,
        "code_quality": {
            "score": code_score,
            "issue_count": len(code_issues),
            "issues": code_issues,
        },
        "data_quality": {
            "datasets": datasets,
            "project_issues": [
                issue for entry in data_results or [] for issue in entry.get("project_issues", [])
            ],
        },
        "ai_review": agent_report or "本次分析未生成 Qwen AI Review；静态分析结论仍然有效。",
        "repair_suggestions": repairs,
        "patch_text": patch_text or "",
        "overall": health,
        "analysis_facts": dict(analysis_facts or {}),
    }


def build_markdown_report(report):
    """Render a GitHub-friendly report from canonical report data."""
    overview = report["project_overview"]
    code = report["code_quality"]
    datasets = report["data_quality"]["datasets"]
    libraries = "、".join(overview.get("main_libraries", [])) or "未识别"
    lines = [
        "# CodeData-Copilot Analysis Report",
        "",
        f"> Generated: {report['generated_at']}",
        "",
        "## 1. Project Overview",
        "",
        f"- 项目名称：{overview.get('project_name', 'Project')}",
        f"- 项目类型/描述：{overview.get('project_type', 'Data Science Project')}；{overview.get('summary', '')}",
        f"- 文件数量：{overview.get('file_count', 0)}",
        f"- 代码文件数量：{overview.get('python_file_count', 0)}",
        f"- 数据文件数量：{overview.get('data_file_count', 0)}",
        f"- 主要依赖/主要库：{libraries}",
        "",
        "## 2. Code Quality Analysis",
        "",
        f"- 代码质量评分：{code['score']}/100",
        f"- 问题数量：{code['issue_count']}",
        "",
    ]
    facts = report.get("analysis_facts", {})
    flow = facts.get("data_flow", {})
    task = facts.get("task_type", {})
    coverage = facts.get("metric_coverage", {})
    model_features = facts.get("model_features", [])
    lines.extend([
        "### 2.0 Data Flow Summary",
        "",
        f"- 数据流：{flow_summary_text(flow)}",
        f"- 建模字段：{', '.join(model_features) if model_features else 'Unable to verify automatically.'}",
        f"- 数据划分：{'Detected' if facts.get('detected_split') else 'Unable to verify automatically.'}",
        f"- 任务类型：{task.get('type', 'Unknown')}（{task.get('confidence', 'Low')} confidence）",
        f"- 检测指标：{', '.join(facts.get('detected_metrics', [])) or 'Unable to verify automatically.'}",
        f"- 指标覆盖：{coverage.get('status', 'Unable to verify')}",
        "",
    ])
    artifacts = facts.get("artifact_analysis", {})
    if artifacts.get("groups"):
        lines.extend(["### 2.0a Duplicate / Near-Duplicate Analysis Artifacts", ""])
        for group in artifacts["groups"]:
            lines.extend([
                f"- {group.get('type')}（similarity {group.get('similarity', 1):.2%}）",
                f"  - Primary source: `{group.get('primary')}`",
                f"  - Files: {', '.join(group.get('files', []))}",
                "  - " + "；".join(
                    f"`{name}` → Duplicate of `{group.get('primary')}`"
                    for name in group.get("files", [])[1:]
                ),
                "  - Repeated matching findings count once; unique findings remain visible.",
            ])
        lines.append("")

    split_roles = facts.get("split_role_analysis", {})
    if split_roles:
        role_details = split_roles.get("split_roles", {})
        lines.extend([
            "### 2.0b Split Role Analysis",
            "",
            f"- Train detected: {split_roles.get('train_detected', False)}",
            f"- Validation detected: {split_roles.get('validation_detected', False)}",
            f"- Test detected: {split_roles.get('test_detected', False)}",
            "- Roles: " + "; ".join(
                f"{role}={', '.join(details.get('variables', [])) or 'not detected'}"
                for role, details in role_details.items()
            ),
        ])
        for risk in split_roles.get("risks", []):
            lines.append(f"- {risk.get('severity', 'Low')} · {risk.get('type')}: {risk.get('evidence')}")
        lines.append("")

    fairness = facts.get("fairness", {})
    if fairness.get("evaluation_exists"):
        lines.extend([
            "### 2.0c Fairness Evaluation",
            "",
            f"- Coverage: {fairness.get('coverage')}",
            "- Dataset fairness metrics: " + (
                ", ".join(item.get("method", "") for item in fairness.get("dataset_metrics", [])) or "None detected"
            ),
            "- Prediction fairness metrics: " + (
                ", ".join(item.get("method", "") for item in fairness.get("prediction_metrics", [])) or "None detected"
            ),
            "- Mitigation: " + (
                ", ".join(item.get("type", "") for item in fairness.get("mitigations", [])) or "None detected"
            ),
            f"- Fairness Before / After Comparison: {fairness.get('before_after_comparison', False)}",
            "",
        ])
    if code["issues"]:
        for index, issue in enumerate(code["issues"], 1):
            location = issue["file"]
            if issue.get("cell"):
                location += f" · Cell {issue['cell']}"
            if issue.get("line"):
                location += f" · Line {issue['line']}"
            lines.extend([
                f"### 2.{index} {issue['type']}",
                "",
                f"- 严重程度：{issue['severity']}",
                f"- 置信度：{issue.get('confidence', 'Medium')}",
                f"- 类别：{issue.get('category', 'Code Quality')}",
                f"- 文件/位置：`{location}`",
                f"- Affected Files ({len(issue.get('affected_files', []))})：{', '.join(issue.get('affected_files', []))}",
                f"- 证据：{issue.get('evidence') or '见定位代码'}",
                f"- 原因：{issue['reason']}",
                f"- 建议：{issue['recommendation']}",
                "",
            ])
            if issue.get("source_snippet"):
                lines.extend(["```python", issue["source_snippet"], "```", ""])
    else:
        lines.extend(["当前静态规则下未发现明显代码问题。", ""])

    lines.extend(["## 3. Data Quality Analysis", ""])
    if datasets:
        for index, dataset in enumerate(datasets, 1):
            shape = dataset["shape"]
            missing = dataset["missing_values"]
            missing_text = "；".join(f"{key}: {value:.2%}" for key, value in missing.items()) or "无"
            dtype_text = "；".join(f"{key}: {value}" for key, value in dataset["data_types"].items()) or "无"
            correlation_text = "；".join(
                f"{item['features']}: {item['value']:.3f}" for item in dataset["high_correlations"]
            ) or "未发现 |r| >= 0.95 的特征对"
            lines.extend([
                f"### 3.{index} {dataset['file']}",
                "",
                f"- 数据规模：{shape.get('rows', 0)} 行 × {shape.get('columns', 0)} 列",
                f"- 缺失情况：{missing_text}",
                f"- 重复数据：{dataset['duplicate_rows']} 行（{dataset['duplicate_rate']:.2%}）",
                f"- 类型情况：{dtype_text}",
                f"- 高相关特征：{correlation_text}",
                f"- 数据质量评分：{dataset['quality_score']}/100",
                f"- 质量问题：{len(dataset.get('issues', []))} 条",
                "",
            ])
            priority = [
                issue for issue in dataset.get("issues", [])
                if issue.get("severity") in {"High", "Medium"}
            ]
            for issue in priority[:20]:
                column = f" · {issue.get('column')}" if issue.get("column") else ""
                lines.extend([
                    f"- **{issue.get('severity')} · {issue.get('type')}**{column}",
                    f"  - Nature / Action: {issue.get('issue_nature', 'Needs Review')} / {issue.get('action', 'Review')}",
                    f"  - Origin: {issue.get('issue_origin', 'Analyzer rule')}",
                    f"  - Evidence: {issue.get('evidence') or issue.get('detail', '')}",
                    f"  - Affected rows: {issue.get('affected_rows') or issue.get('count') or 'Not quantified'}",
                    f"  - Recommendation: {issue.get('recommendation', '')}",
                ])
            low_counts = {}
            for issue in dataset.get("issues", []):
                if issue.get("severity") == "Low":
                    label = f"{issue.get('category', 'Other')} / {issue.get('type', 'Finding')}"
                    low_counts[label] = low_counts.get(label, 0) + 1
            if low_counts:
                summary = "；".join(f"{label} {count}" for label, count in sorted(low_counts.items()))
                lines.extend([f"- Low 级提示汇总：{summary}", ""])
    else:
        lines.extend(["项目中没有可分析的 CSV 文件。", ""])

    project_data_issues = report["data_quality"].get("project_issues", [])
    if project_data_issues:
        lines.extend(["### 3.x Multi-file Consistency", ""])
        for issue in project_data_issues:
            lines.append(f"- {issue.get('type')}：{issue.get('evidence')}")
        lines.append("")

    impacts = facts.get("quantified_impacts", [])
    if impacts:
        lines.extend(["### 3.y Data Impact", ""])
        for impact in impacts:
            if impact.get("quantification") == "Exact":
                lines.extend([
                    f"- `{impact.get('operation')}`：Exact",
                    f"  - Before: {impact.get('before')} rows",
                    f"  - After: {impact.get('after')} rows",
                    f"  - Removed: {impact.get('removed')} rows（{impact.get('removal_rate', 0):.2%}）",
                    f"  - Relevant columns: {', '.join(impact.get('relevant_columns', [])) or 'None detected'}",
                    f"  - Non-model columns: {', '.join(impact.get('non_model_columns', [])) or 'None'}",
                ])
            else:
                lines.append(
                    f"- `{impact.get('operation')}`：Unable to quantify（{impact.get('reason', '')}）"
                )
        lines.append("")

    model_feature_set = set(facts.get("model_features", []))
    target_column = (facts.get("target") or {}).get("column")
    non_model_missing = sorted({
        column
        for dataset in datasets
        for column in dataset.get("missing_values", {})
        if column not in model_feature_set and column != target_column
    })
    if non_model_missing and model_feature_set:
        lines.extend([
            "- 未进入当前模型输入的缺失字段："
            + ", ".join(non_model_missing)
            + "；这些字段不会直接构成当前模型输入的缺失影响。",
            "",
        ])

    duplicate_analysis = facts.get("duplicate_split_analysis", {})
    if duplicate_analysis:
        lines.extend(["### 3.z Duplicate Split Risk", ""])
        if duplicate_analysis.get("status") == "Exact":
            lines.extend([
                f"- Duplicate groups: {duplicate_analysis.get('duplicate_groups', 0)}",
                f"- Cross-split duplicate groups: {duplicate_analysis.get('cross_split_duplicate_groups', 0)}",
                f"- Train/Test shared records: {duplicate_analysis.get('shared_records', 0)}",
            ])
        else:
            lines.append(
                "- Detected duplicate records, but the exact train/test overlap could not be reconstructed. "
                + str(duplicate_analysis.get("reason") or "")
            )
        lines.append("")

    all_data_issues = [issue for dataset in datasets for issue in dataset.get("issues", [])]
    confirmed = [issue for issue in all_data_issues if issue.get("issue_nature") in {"Confirmed Invalid", "Rule Violation"} and issue.get("confidence") == "High"]
    potential = [issue for issue in all_data_issues if issue.get("confidence") in {"Medium", "Low"}]
    informational = [issue for issue in all_data_issues if issue.get("action") == "Informational"]
    lines.extend([
        "### 3.w Confirmed vs Potential Data Problems",
        "",
        f"- Confirmed: {len(confirmed)}",
        f"- Potential / Needs Review: {len(potential)}",
        f"- Informational: {len(informational)}",
        "",
    ])

    lines.extend([
        "## 4. AI Review",
        "",
        report["ai_review"].strip(),
        "",
        "## 5. Repair Suggestions",
        "",
    ])
    repairs = report["repair_suggestions"]
    if repairs:
        for index, repair in enumerate(repairs, 1):
            lines.extend([
                f"### 5.{index} {repair.get('problem', 'Repair')}",
                "",
                f"- 位置：`{repair.get('location', 'Unknown')}`",
                f"- 修改说明：{repair.get('explanation', repair.get('reason', ''))}",
                f"- 验证状态：{repair.get('validation_status', 'Not Automatically Verifiable')}",
                f"- 验证说明：{repair.get('validation_detail', '未执行自动验证。')}",
                "",
                "**原代码**",
                "",
                "```python",
                repair.get("before_code") or "# 未提供原始代码片段，无法生成可应用的 diff。",
                "```",
                "",
                "**建议修改代码**",
                "",
                "```python",
                repair.get("after_code") or "# 暂无建议代码",
                "```",
                "",
            ])
    else:
        lines.extend(["当前规则下没有可生成的确定性修复建议。", ""])
    if report["patch_text"]:
        lines.extend(["### Patch Information", "", "```diff", report["patch_text"].rstrip(), "```", ""])
    else:
        lines.extend(["Patch：未生成（可能缺少可逐字引用的原始代码片段）。", ""])

    score = report["overall"].get("score", 0)
    caveat = report["overall"].get("caveat", "启发式静态评分。")
    lines.extend([
        "## 6. Overall Score / Summary",
        "",
        f"- 综合评分：**{score}/100**",
        f"- 综合评价：{overview.get('summary', '')}",
        f"- 说明：{caveat}",
        "",
    ])
    return "\n".join(lines)


def _plain_markdown(text):
    text = re.sub(r"`([^`]*)`", r"\1", str(text))
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text)
    return text.strip()


def build_pdf_report(report):
    """Render a clean, in-memory PDF using ReportLab's CJK CID font."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import (
            KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
            Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("PDF 报告依赖 reportlab，请安装 requirements.txt 后重试。") from exc

    font_name = "STSong-Light"
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=report["title"],
        author="CodeData-Copilot",
    )
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "CJKBody", parent=base["BodyText"], fontName=font_name, fontSize=9.2,
        leading=14, textColor=colors.HexColor("#293241"), spaceAfter=5,
    )
    title_style = ParagraphStyle(
        "CJKTitle", parent=body, fontName="Helvetica-Bold", fontSize=22,
        leading=28, alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"), spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=body, fontSize=9, leading=13, alignment=TA_CENTER,
        textColor=colors.HexColor("#64748B"), spaceAfter=18,
    )
    heading = ParagraphStyle(
        "Heading", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=19,
        textColor=colors.HexColor("#2563EB"), spaceBefore=12, spaceAfter=8,
    )
    subheading = ParagraphStyle(
        "Subheading", parent=body, fontSize=11, leading=16,
        textColor=colors.HexColor("#0F172A"), spaceBefore=8, spaceAfter=5,
    )
    label = ParagraphStyle(
        "Label", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#64748B"),
    )
    code_style = ParagraphStyle(
        "Code", parent=body, fontName="Courier", fontSize=7.4, leading=10,
        leftIndent=5, rightIndent=5, borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#F8FAFC"),
        spaceBefore=4, spaceAfter=7,
    )

    def mixed_markup(value):
        """Use built-in Latin fonts for ASCII and the CJK font for Chinese."""
        escaped_lines = []
        for line in str(value).splitlines() or [""]:
            escaped_line = escape(line)
            escaped_line = re.sub(
                r"([\x20-\x7E]+)",
                lambda match: f'<font name="Helvetica">{match.group(1)}</font>',
                escaped_line,
            )
            escaped_lines.append(escaped_line)
        return "<br/>".join(escaped_lines)

    def p(value, style=body):
        return Paragraph(mixed_markup(value), style)

    def section(number, text):
        return Paragraph(f"{number}. {escape(text)}", heading)

    overview = report["project_overview"]
    story = [
        Paragraph(report["title"], title_style),
        p(f"{overview.get('project_name', 'Project')} · {report['generated_at']}", subtitle),
        section(1, "Project Overview"),
    ]
    overview_rows = [
        [p("项目名称", label), p(overview.get("project_name", "Project"))],
        [p("项目类型", label), p(overview.get("project_type", "Data Science Project"))],
        [p("文件统计", label), p(
            f"{overview.get('file_count', 0)} files · {overview.get('python_file_count', 0)} code · "
            f"{overview.get('data_file_count', 0)} data"
        )],
        [p("主要依赖", label), p("、".join(overview.get("main_libraries", [])) or "未识别")],
        [p("项目描述", label), p(overview.get("summary", ""))],
    ]
    overview_table = Table(overview_rows, colWidths=[31 * mm, 123 * mm])
    overview_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([overview_table, Spacer(1, 5 * mm), section(2, "Code Quality Analysis")])
    code = report["code_quality"]
    summary_table = Table([
        [p("代码质量评分", label), p(f"{code['score']}/100"), p("问题数量", label), p(code["issue_count"])],
    ], colWidths=[31 * mm, 46 * mm, 31 * mm, 46 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    facts = report.get("analysis_facts", {})
    flow = facts.get("data_flow", {})
    task = facts.get("task_type", {})
    coverage = facts.get("metric_coverage", {})
    story.extend([
        p("Data Flow Summary", subheading),
        p(f"数据流：{flow_summary_text(flow)}"),
        p(
            "建模字段：" + (
                "、".join(facts.get("model_features", []))
                or "Unable to verify automatically."
            )
        ),
        p(
            f"任务类型：{task.get('type', 'Unknown')}（{task.get('confidence', 'Low')} confidence）；"
            f"指标：{', '.join(facts.get('detected_metrics', [])) or 'Unable to verify automatically.'}；"
            f"覆盖：{coverage.get('status', 'Unable to verify')}"
        ),
    ])
    artifacts = facts.get("artifact_analysis", {})
    if artifacts.get("groups"):
        story.append(p("Duplicate / Near-Duplicate Analysis Artifacts", subheading))
        for group in artifacts["groups"]:
            story.append(p(
                f"{group.get('type')} · Primary={group.get('primary')} · "
                f"Affected Files ({len(group.get('files', []))})="
                f"{', '.join(group.get('files', []))} · similarity={group.get('similarity', 1):.2%}"
            ))
    split_roles = facts.get("split_role_analysis", {})
    if split_roles:
        story.extend([
            p("Split Role Analysis", subheading),
            p(
                f"train={split_roles.get('train_detected', False)}；"
                f"validation={split_roles.get('validation_detected', False)}；"
                f"test={split_roles.get('test_detected', False)}"
            ),
        ])
        for risk in split_roles.get("risks", []):
            story.append(p(f"{risk.get('severity')} · {risk.get('type')}：{risk.get('evidence')}"))
    fairness = facts.get("fairness", {})
    if fairness.get("evaluation_exists"):
        story.extend([
            p("Fairness Evaluation", subheading),
            p(f"Coverage：{fairness.get('coverage')}"),
            p(
                "Dataset metrics："
                + (", ".join(item.get("method", "") for item in fairness.get("dataset_metrics", [])) or "None")
            ),
            p(
                "Prediction metrics："
                + (", ".join(item.get("method", "") for item in fairness.get("prediction_metrics", [])) or "None")
            ),
            p(f"Fairness Before / After Comparison：{fairness.get('before_after_comparison', False)}"),
        ])
    if code["issues"]:
        for issue in code["issues"]:
            location = issue["file"]
            if issue.get("cell"):
                location += f" · Cell {issue['cell']}"
            if issue.get("line"):
                location += f" · Line {issue['line']}"
            block = [
                p(
                    f"{issue['severity']} · {issue.get('confidence', 'Medium')} confidence · "
                    f"{issue.get('category', 'Code Quality')} · {issue['type']}",
                    subheading,
                ),
                p(f"位置：{location}"),
                p(
                    f"Affected Files ({len(issue.get('affected_files', []))})："
                    f"{', '.join(issue.get('affected_files', []))}"
                ),
                p(f"证据：{issue.get('evidence') or '见定位代码'}"),
                p(f"原因：{issue['reason']}"),
                p(f"建议：{issue['recommendation']}"),
            ]
            if issue.get("source_snippet"):
                block.append(Preformatted(issue["source_snippet"], code_style))
            story.append(KeepTogether(block))
    else:
        story.append(p("当前静态规则下未发现明显代码问题。"))

    story.append(section(3, "Data Quality Analysis"))
    datasets = report["data_quality"]["datasets"]
    if datasets:
        for dataset in datasets:
            shape = dataset["shape"]
            missing = "；".join(
                f"{name}: {value:.2%}" for name, value in dataset["missing_values"].items()
            ) or "无"
            correlations = "；".join(
                f"{item['features']}: {item['value']:.3f}" for item in dataset["high_correlations"]
            ) or "未发现 |r| >= 0.95 的特征对"
            story.extend([
                p(dataset["file"], subheading),
                p(f"规模：{shape.get('rows', 0)} 行 × {shape.get('columns', 0)} 列"),
                p(f"缺失：{missing}"),
                p(f"重复：{dataset['duplicate_rows']} 行（{dataset['duplicate_rate']:.2%}）"),
                p(f"高相关特征：{correlations}"),
                p(f"数据质量评分：{dataset['quality_score']}/100"),
                p(f"质量问题：{len(dataset.get('issues', []))} 条"),
            ])
            for issue in [
                item for item in dataset.get("issues", [])
                if item.get("severity") in {"High", "Medium"}
            ][:15]:
                column = f" · {issue.get('column')}" if issue.get("column") else ""
                story.append(p(
                    f"{issue.get('severity')} · {issue.get('type')}{column}："
                    f"{issue.get('evidence') or issue.get('detail', '')}；"
                    f"Nature={issue.get('issue_nature', 'Needs Review')}；"
                    f"Action={issue.get('action', 'Review')}；"
                    f"Origin={issue.get('issue_origin', 'Analyzer rule')}"
                ))
            low_counts = {}
            for issue in dataset.get("issues", []):
                if issue.get("severity") == "Low":
                    label_text = f"{issue.get('category', 'Other')} / {issue.get('type', 'Finding')}"
                    low_counts[label_text] = low_counts.get(label_text, 0) + 1
            if low_counts:
                summary = "；".join(
                    f"{label_text} {count}" for label_text, count in sorted(low_counts.items())
                )
                story.append(p(f"Low 级提示汇总：{summary}"))
        project_data_issues = report["data_quality"].get("project_issues", [])
        if project_data_issues:
            story.append(p("Multi-file Consistency", subheading))
            for issue in project_data_issues[:15]:
                story.append(p(
                    f"{issue.get('severity', 'Low')} · {issue.get('type', 'Schema Difference')}："
                    f"{issue.get('evidence', '')}"
                ))
        impacts = facts.get("quantified_impacts", [])
        if impacts:
            story.append(p("Data Impact", subheading))
            for impact in impacts:
                if impact.get("quantification") == "Exact":
                    story.append(p(
                        f"{impact.get('operation')} · Exact：Before {impact.get('before')} rows；"
                        f"After {impact.get('after')} rows；Removed {impact.get('removed')} "
                        f"({impact.get('removal_rate', 0):.2%})；Relevant columns: "
                        f"{', '.join(impact.get('relevant_columns', [])) or 'None detected'}"
                    ))
                else:
                    story.append(p(
                        f"{impact.get('operation')}：Unable to quantify。{impact.get('reason', '')}"
                    ))
        model_feature_set = set(facts.get("model_features", []))
        target_column = (facts.get("target") or {}).get("column")
        non_model_missing = sorted({
            column for dataset in datasets for column in dataset.get("missing_values", {})
            if column not in model_feature_set and column != target_column
        })
        if non_model_missing and model_feature_set:
            story.append(p(
                "未进入当前模型输入的缺失字段：" + "、".join(non_model_missing)
                + "；不会直接构成当前模型输入的缺失影响。"
            ))
        duplicate_analysis = facts.get("duplicate_split_analysis", {})
        if duplicate_analysis:
            story.append(p("Duplicate Split Risk", subheading))
            if duplicate_analysis.get("status") == "Exact":
                story.append(p(
                    f"Duplicate groups={duplicate_analysis.get('duplicate_groups', 0)}；"
                    f"Cross-split groups={duplicate_analysis.get('cross_split_duplicate_groups', 0)}；"
                    f"Shared records={duplicate_analysis.get('shared_records', 0)}"
                ))
            else:
                story.append(p(
                    "Detected duplicate records, but the exact train/test overlap could not be reconstructed. "
                    + str(duplicate_analysis.get("reason") or "")
                ))
    else:
        story.append(p("项目中没有可分析的 CSV 文件。"))

    story.extend([PageBreak(), section(4, "AI Review")])
    in_code = False
    code_lines = []
    for raw_line in report["ai_review"].splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines) or " ", code_style))
                code_lines = []
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(raw_line)
        elif stripped:
            clean = _plain_markdown(stripped.lstrip("- "))
            style = subheading if stripped.startswith("#") else body
            story.append(p(clean, style))
    if code_lines:
        story.append(Preformatted("\n".join(code_lines), code_style))

    story.append(section(5, "Repair Suggestions"))
    repairs = report["repair_suggestions"]
    if repairs:
        for repair in repairs:
            story.extend([
                p(repair.get("problem", "Repair"), subheading),
                p(f"位置：{repair.get('location', 'Unknown')}"),
                p(f"修改说明：{repair.get('explanation', repair.get('reason', ''))}"),
                p(f"验证状态：{repair.get('validation_status', 'Not Automatically Verifiable')}"),
                p("Before", label),
                (
                    Preformatted(repair.get("before_code"), code_style)
                    if repair.get("before_code")
                    else p("未提供原始代码片段，无法生成可应用的 diff。")
                ),
                p("After", label),
                Preformatted(repair.get("after_code") or "暂无建议代码", code_style),
            ])
    else:
        story.append(p("当前规则下没有可生成的确定性修复建议。"))
    if report["patch_text"]:
        story.extend([
            p("Patch Information", subheading),
            Preformatted(report["patch_text"], code_style),
        ])

    story.extend([PageBreak(), section(6, "Overall Score / Summary")])
    score = report["overall"].get("score", 0)
    story.extend([
        Paragraph(f"{score}/100", ParagraphStyle(
            "Score", parent=title_style, alignment=TA_LEFT, fontSize=25,
            textColor=colors.HexColor("#2563EB"), spaceAfter=5,
        )),
        p(overview.get("summary", "")),
        p(report["overall"].get("caveat", "启发式静态评分。"), label),
    ])

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, 10 * mm, "CodeData-Copilot · Static project audit")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
