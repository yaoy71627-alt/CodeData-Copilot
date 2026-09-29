import ast
from datetime import datetime
import html
from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from agent import (
    build_repair_suggestions,
    load_knowledge_documents,
    retrieve_relevant_knowledge,
    run_agent,
)
from code_analyzer import (
    analyze_code_with_ast,
    analyze_notebook,
    read_notebook,
    redact_sensitive_code,
)
from data_analyzer import analyze_csv, analyze_dataset_consistency
from data_flow_analyzer import (
    analyze_notebook_data_flow,
    analyze_python_data_flow,
    enrich_model_features,
    flow_summary_text,
    merge_data_flows,
    quantify_data_impacts,
    reconcile_code_issues_with_flow,
)
from consistency_validator import (
    build_analysis_facts,
    validate_agent_consistency,
    validate_report_consistency,
)
from file_loader import decode_python_file, materialize_project_uploads, materialize_uploads
from github_loader import load_github_project
from patch_generator import build_unified_patch
from repair_validator import validate_repair_suggestions
from project_overview import build_project_overview
from project_score import calculate_project_score
from project_dedup import detect_duplicate_artifacts, consolidate_duplicate_code_issues
from report import (
    build_data_issues,
    build_markdown_report,
    build_pdf_report,
    build_report_data,
    summarize_results,
)


st.set_page_config(
    page_title="CodeData-Copilot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLE_PATH = Path(__file__).with_name("style.css")
if STYLE_PATH.is_file():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state():
    defaults = {
        "code_result": None,
        "data_result": None,
        "agent_report": None,
        "analysis_error": None,
        "analyzed_files": None,
        "data_results": [],
        "data_consistency_issues": [],
        "data_flow": None,
        "analysis_facts": None,
        "quantified_impacts": [],
        "artifact_analysis": {"groups": [], "files": []},
        "health": None,
        "analysis_warnings": [],
        "project_stats": None,
        "project_context": None,
        "skipped_files": [],
        "import_source": None,
        "project_overview": None,
        "repair_suggestions": [],
        "patch_text": "",
        "report_data": None,
        "report_markdown": None,
        "report_pdf": None,
        "report_error": None,
        "last_analyzed_at": None,
        "active_page": "Home",
        "import_mode": "Local Project",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def read_csv_preview(uploaded_file):
    content = uploaded_file.getvalue()
    try:
        return pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(content), encoding="gb18030")


def original_evidence(source, issue_type):
    """Attach only an actual source expression and its real line number."""
    target = "dropna" if issue_type == "Missing Value Handling Risk" else "fit_transform"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"source_snippet": None, "line": None}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == target):
            return {
                "source_snippet": ast.get_source_segment(source, node),
                "line": getattr(node, "lineno", None),
            }
    return {"source_snippet": None, "line": None}


def original_snippet(source, issue_type):
    """Compatibility wrapper used by existing tests and extensions."""
    return original_evidence(source, issue_type)["source_snippet"]


def read_text_file(path):
    """Decode Markdown and other repository text without executing it."""
    content = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码无法识别，请转换为 UTF-8 后重试")


def limited_excerpt(text, limit=3000):
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n…（内容已截断）"


ANALYSIS_STAGES = (
    "正在解析项目",
    "正在扫描代码质量",
    "正在分析数据质量",
    "正在调用知识库",
    "正在进行 AI Review",
    "正在生成修复建议",
    "分析完成",
)

ANALYSIS_STAGE_LABELS = (
    "Project parsed",
    "Code scan",
    "Data quality analysis",
    "Knowledge enhancement",
    "AI Review",
    "Report generation",
    "Analysis complete",
)


def analyze_project_files(files, generate_ai=True, project_metadata=None, progress_callback=None):
    """Run the existing analyzers in observable stages and persist result artifacts."""
    def notify(index, state="running", detail=None):
        if progress_callback:
            progress_callback(index, state, detail)

    code_result, data_results, sources, warnings = [], [], [], []
    file_flows = []
    code_names, data_names, documentation_names = [], [], []
    documentation, code_context, parsed_code, csv_files = [], [], [], []
    skipped_files = list(getattr(files, "skipped_files", []))
    files = list(files)
    artifact_analysis = detect_duplicate_artifacts(files)

    notify(0, "running")
    for name, path in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                csv_files.append((name, path))
            elif suffix == ".ipynb":
                cells = read_notebook(str(path))
                sources.extend(cells)
                code_names.append(name)
                parsed_code.append((name, "notebook", path))
                safe_cells = [redact_sensitive_code(cell) for cell in cells]
                code_context.append({"file": name, "excerpt": limited_excerpt("\n\n".join(safe_cells))})
            elif suffix == ".py":
                source = decode_python_file(path.read_bytes())
                sources.append(source)
                code_names.append(name)
                parsed_code.append((name, "python", source))
                code_context.append({"file": name, "excerpt": limited_excerpt(redact_sensitive_code(source))})
            elif suffix == ".md":
                documentation_names.append(name)
                documentation.append({"file": name, "excerpt": limited_excerpt(read_text_file(path), 4000)})
        except Exception as exc:
            warnings.append(f"{name}：解析失败（{exc}）。")
    notify(0, "complete", f"已识别 {len(files)} 个候选文件")

    notify(1, "running")
    for name, kind, content in parsed_code:
        try:
            if kind == "python":
                issues = analyze_code_with_ast(content)
                file_flows.append(analyze_python_data_flow(content, file_name=name))
                if issues:
                    code_result.append({"file": name, "issues": issues})
            else:
                file_flows.append(analyze_notebook_data_flow(str(content)))
                for notebook_result in analyze_notebook(str(content)):
                    code_result.append({"file": name, **notebook_result})
        except Exception as exc:
            warnings.append(f"{name}：代码质量分析失败（{exc}）。")
    notify(1, "complete", f"完成 {len(code_names)} 个代码文件的静态检查")

    data_flow = merge_data_flows(file_flows)
    notify(2, "running")
    for name, path in csv_files:
        try:
            data_report = analyze_csv(str(path), data_flow=data_flow)
            if not data_report["shape"]["rows"] or not data_report["shape"]["columns"]:
                warnings.append(f"{name}：数据集为空，评分会反映这一情况。")
            data_results.append({"file": name, "result": data_report})
            data_names.append(name)
        except Exception as exc:
            warnings.append(f"{name}：数据质量分析失败（{exc}）。")
    data_consistency_issues = analyze_dataset_consistency(data_results)
    if data_results and data_consistency_issues:
        data_results[0]["project_issues"] = data_consistency_issues
    if data_results:
        data_flow = enrich_model_features(data_flow, data_results[0]["result"])
    code_result = reconcile_code_issues_with_flow(code_result, data_flow)
    code_result = consolidate_duplicate_code_issues(code_result, artifact_analysis)
    quantified_impacts = []
    if csv_files and data_results:
        quantified_impacts = quantify_data_impacts(
            data_flow, csv_files[0][1], data_results[0]["result"]
        )
        data_results[0]["result"]["quantified_impacts"] = quantified_impacts
    analysis_facts = build_analysis_facts(
        code_result,
        data_results,
        data_flow,
        quantified_impacts=quantified_impacts,
        duplicate_split_analysis=(
            data_results[0]["result"].get("duplicate_split_analysis", {})
            if data_results else {}
        ),
        business_rule_results=(
            data_results[0]["result"].get("business_rule_results", [])
            if data_results else []
        ),
        artifact_analysis=artifact_analysis,
    )
    notify(2, "complete", f"完成 {len(data_results)} 个数据文件的质量检查")

    if not code_names and not data_results and not documentation_names:
        raise ValueError("没有成功分析的文件。请检查文件格式与内容。")
    health = calculate_project_score(code_result, data_results, code_files_count=len(code_names))
    overview = build_project_overview(
        [name for name, _ in files], sources, data_results, project_metadata or {}
    )
    project_context = {
        "project_information": project_metadata or {},
        "project_overview": overview,
        "project_files": [name for name, _ in files],
        "code_files": code_names,
        "data_files": data_names,
        "data_consistency_issues": data_consistency_issues,
        "analysis_facts": analysis_facts,
        "artifact_analysis": artifact_analysis,
        "documentation_files": documentation_names,
        "documentation": documentation,
        "user_code_context": code_context,
    }

    notify(3, "running")
    relevant_knowledge = retrieve_relevant_knowledge(code_result, data_results, project_context)
    notify(3, "complete", f"已选择 {relevant_knowledge.count('- ')} 条相关审查准则")

    agent_report = None
    agent_error = None
    notify(4, "running")
    if generate_ai:
        try:
            agent_report = run_agent(
                code_result,
                data_results,
                project_context,
                knowledge_context=relevant_knowledge,
                analysis_facts=analysis_facts,
            )
            agent_report, consistency_notes = validate_agent_consistency(
                agent_report, analysis_facts
            )
            warnings.extend(consistency_notes)
            notify(4, "complete", "Qwen Review 已生成")
        except Exception as exc:
            agent_error = str(exc)
            notify(4, "error", "AI Review 未生成，基础分析仍可使用")
    else:
        notify(4, "complete", "本次未启用 AI Review")

    notify(5, "running")
    repair_suggestions = build_repair_suggestions(code_result, data_flow, data_results)
    repair_suggestions = validate_repair_suggestions(
        repair_suggestions, data_flow, data_results
    )
    patch_text = build_unified_patch(repair_suggestions)
    report_data = build_report_data(
        overview,
        code_result,
        data_results,
        agent_report,
        repair_suggestions,
        patch_text,
        health,
        analysis_facts=analysis_facts,
    )
    report_data = validate_report_consistency(report_data)
    report_markdown = build_markdown_report(report_data)
    report_pdf = None
    report_error = None
    try:
        report_pdf = build_pdf_report(report_data)
    except Exception as exc:
        report_error = str(exc)
        warnings.append(f"PDF 报告生成失败：{exc}")
    notify(5, "complete", f"已生成 {len(repair_suggestions)} 条修复建议和报告")

    st.session_state.code_result = code_result
    st.session_state.data_results = data_results
    st.session_state.data_consistency_issues = data_consistency_issues
    st.session_state.data_flow = data_flow
    st.session_state.analysis_facts = analysis_facts
    st.session_state.quantified_impacts = quantified_impacts
    st.session_state.artifact_analysis = artifact_analysis
    st.session_state.data_result = data_results[0]["result"] if data_results else None
    st.session_state.agent_report = agent_report
    st.session_state.analysis_error = agent_error
    st.session_state.analysis_warnings = warnings
    st.session_state.health = health
    st.session_state.project_overview = overview
    st.session_state.project_context = project_context
    st.session_state.project_stats = {
        "files": overview["file_count"],
        "code_files": overview["python_file_count"],
        "data_files": overview["data_file_count"],
        "documentation_files": overview["documentation_file_count"],
    }
    st.session_state.skipped_files = skipped_files
    st.session_state.analyzed_files = {
        "code": f"{len(code_names)} 个代码文件",
        "data": f"{len(data_results)} 个 CSV 文件",
        "documentation": f"{len(documentation_names)} 个 Markdown 文件",
    }
    st.session_state.import_source = (project_metadata or {}).get("source", "Analysis pipeline")
    st.session_state.repair_suggestions = repair_suggestions
    st.session_state.patch_text = patch_text
    st.session_state.report_data = report_data
    st.session_state.report_markdown = report_markdown
    st.session_state.report_pdf = report_pdf
    st.session_state.report_error = report_error
    st.session_state.last_analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    notify(6, "complete")


def run_full_analysis(code_file, data_file, generate_ai=True, progress_callback=None):
    """Keep the original local-upload route and its validation."""
    with materialize_uploads(code_file, data_file) as files:
        analyze_project_files([
            (code_file.name, files["code_path"]),
            (data_file.name, files["data_path"]),
        ], generate_ai, {"source": "Local upload"}, progress_callback)
    st.session_state.import_source = "Local upload"


def run_uploaded_project(uploaded_files, generate_ai=True, progress_callback=None):
    """Analyze a local multi-file project without changing the legacy upload route."""
    with materialize_project_uploads(uploaded_files) as files:
        analyze_project_files(
            files, generate_ai, {"source": "Local upload"}, progress_callback
        )


def run_with_analysis_status(operation):
    """Render a concise task checklist while the supplied analysis operation runs."""
    states = ["pending"] * len(ANALYSIS_STAGES)
    details = [None] * len(ANALYSIS_STAGES)
    current = {"index": 0}

    with st.status("正在准备分析", expanded=True) as status:
        progress = st.progress(0, text="0%")
        checklist = st.empty()

        def render():
            icons = {"pending": "○", "running": "●", "complete": "✓", "error": "×"}
            if "running" in states:
                pipeline_status = "Running"
            elif all(value == "complete" for value in states):
                pipeline_status = "Complete"
            elif "error" in states:
                pipeline_status = "Needs attention"
            else:
                pipeline_status = "Ready"
            rows = []
            for index, label in enumerate(ANALYSIS_STAGE_LABELS):
                detail = (
                    f'<small>{html.escape(details[index])}</small>'
                    if details[index] else ""
                )
                rows.append(
                    f'<div class="pipeline-row pipeline-{states[index]}">'
                    f'<b>{icons[states[index]]}</b><div><span>{html.escape(label)}</span>{detail}</div></div>'
                )
            checklist.markdown(
                '<div class="pipeline-shell"><div class="pipeline-head">'
                '<div><span>ANALYSIS PIPELINE</span><strong>Execution trace</strong></div>'
                f'<em>{html.escape(pipeline_status)}</em></div>'
                '<div class="pipeline-list">' + "".join(rows) + "</div></div>",
                unsafe_allow_html=True,
            )
            completed = sum(value in {"complete", "error"} for value in states)
            percent = round(completed / len(states) * 100)
            progress.progress(completed / len(states), text=f"{percent}%")

        def callback(index, state="running", detail=None):
            current["index"] = index
            states[index] = state
            details[index] = detail
            render()
            if state == "running":
                status.update(label=ANALYSIS_STAGES[index], state="running", expanded=True)
            elif index == len(ANALYSIS_STAGES) - 1:
                status.update(label="分析完成", state="complete", expanded=False)

        render()
        try:
            operation(callback)
        except Exception:
            index = current["index"]
            callback(index, "error", "此阶段执行失败")
            status.update(label=f"分析在“{ANALYSIS_STAGES[index]}”阶段失败", state="error", expanded=True)
            raise


def rebuild_session_report():
    """Refresh all report formats after an AI Review is regenerated."""
    suggestions = st.session_state.repair_suggestions or build_repair_suggestions(
        st.session_state.code_result,
        st.session_state.data_flow,
        st.session_state.data_results,
    )
    suggestions = validate_repair_suggestions(
        suggestions, st.session_state.data_flow, st.session_state.data_results
    )
    patch_text = st.session_state.patch_text or build_unified_patch(suggestions)
    report_data = build_report_data(
        st.session_state.project_overview,
        st.session_state.code_result,
        st.session_state.data_results,
        st.session_state.agent_report,
        suggestions,
        patch_text,
        st.session_state.health,
        analysis_facts=st.session_state.analysis_facts,
    )
    report_data = validate_report_consistency(report_data)
    st.session_state.report_data = report_data
    st.session_state.report_markdown = build_markdown_report(report_data)
    st.session_state.report_error = None
    try:
        st.session_state.report_pdf = build_pdf_report(report_data)
    except Exception as exc:
        st.session_state.report_pdf = None
        st.session_state.report_error = str(exc)


def require_results():
    if st.session_state.analyzed_files is None:
        st.info("请先在 Project Import 页面上传文件或导入 GitHub 仓库，并运行分析。")
        return False
    return True


def render_summary():
    reports = [item["result"] for item in st.session_state.data_results]
    code_count = summarize_results(st.session_state.code_result, None)["code_issue_count"]
    data_count = sum(len(build_data_issues(result)) for result in reports)
    stats = st.session_state.project_stats or {
        "files": len(reports), "code_files": 0, "data_files": len(reports),
        "documentation_files": 0,
    }
    score = st.session_state.health["score"] if st.session_state.health else 0
    items = (
        ("Overall", f"{score}/100", "primary"),
        ("Code", str(stats["code_files"]), "neutral"),
        ("Data", str(stats["data_files"]), "teal"),
        ("Issues", str(code_count + data_count), "warning"),
        ("Files", str(stats["files"]), "neutral"),
    )
    cells = "".join(
        f'<div class="stat-cell stat-{tone}"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value, tone in items
    )
    st.markdown(f'<div class="stats-bar">{cells}</div>', unsafe_allow_html=True)


def issue_quality_score(issues):
    """Return a compact 0-100 display score without replacing project scoring."""
    weights = {"High": 14, "Medium": 6, "Low": 1.5}
    grouped = {}
    for issue in issues or []:
        key = (issue.get("category", "Quality"), issue.get("rule_id", issue.get("type")))
        grouped.setdefault(key, []).append(weights.get(issue.get("severity", "Low"), 1.5))
    penalty = sum(max(values) + min(2, max(0, len(values) - 1) * 0.5)
                  for values in grouped.values())
    return max(0, round(100 - min(100, penalty)))


def issue_table(issues, include_file=False):
    """Build consistent compact rows for Data Quality and Code Review."""
    rows = []
    for issue in issues or []:
        row = {
            "Severity": str(issue.get("severity", "Low")).upper(),
            "Confidence": issue.get("confidence", "Medium"),
            "Nature": issue.get("issue_nature", "Needs Review"),
            "Origin": issue.get("issue_origin", "Analyzer rule"),
            "Category": issue.get("category", "Quality"),
            "Column": issue.get("column") or "—",
            "Issue": issue.get("type", "Quality issue"),
            "Evidence": issue.get("evidence") or issue.get("message", ""),
            "Affected Rows": (
                ", ".join(map(str, issue.get("affected_rows", [])[:8]))
                if issue.get("affected_rows") else (issue.get("count") if issue.get("count") is not None else "—")
            ),
            "Action": issue.get("action", "Review"),
        }
        if include_file:
            row["File"] = issue.get("file", "—")
            row["Affected Files"] = ", ".join(issue.get("affected_files", [])) or row["File"]
            row["Line"] = issue.get("line") or "—"
            row.pop("Column", None)
        rows.append(row)
    return pd.DataFrame(rows)


def render_project_health_panel():
    """Render the existing static score as a compact project-health focal point."""
    health = st.session_state.health or {}
    if not health:
        return

    score = max(0, min(100, int(health.get("score", 0))))
    breakdown = health.get("breakdown", {})
    code_quality = round(breakdown.get("代码质量", 0) / 35 * 100)
    data_quality = round(breakdown.get("数据质量", 0) / 35 * 100)
    code_quality = max(0, min(100, code_quality))
    data_quality = max(0, min(100, data_quality))

    code_issues = summarize_results(st.session_state.code_result, None)["code_issue_count"]
    repair_count = len(st.session_state.repair_suggestions or [])
    if code_issues == 0 or repair_count >= code_issues:
        repairability, repair_value = "High", 92
    elif repair_count:
        repairability, repair_value = "Medium", 64
    else:
        repairability, repair_value = "Low", 28

    if score >= 80:
        project_status, risk_level, risk_value, risk_tone = "HEALTHY", "Low", 24, "teal"
    elif score >= 60:
        project_status, risk_level, risk_value, risk_tone = "REVIEW", "Medium", 56, "warning"
    else:
        project_status, risk_level, risk_value, risk_tone = "ATTENTION", "High", 84, "danger"

    signals = (
        ("Code Quality", f"{code_quality}", code_quality, "blue"),
        ("Data Quality", f"{data_quality}", data_quality, "teal"),
        ("Risk Level", risk_level, risk_value, risk_tone),
        ("Repairability", repairability, repair_value, "teal"),
    )
    signal_rows = "".join(
        '<div class="health-signal">'
        f'<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        '<div class="health-track">'
        f'<i class="health-fill health-{tone}" style="width:{percent}%"></i></div></div>'
        for label, value, percent, tone in signals
    )
    caveat = html.escape(health.get("caveat", "Static analysis score"))
    st.markdown(
        '<div class="project-health-panel">'
        '<div class="health-score-block">'
        '<div class="health-panel-head"><span>PROJECT HEALTH</span>'
        f'<em class="health-status health-status-{risk_tone}">{project_status}</em></div>'
        '<div class="health-score-label">Overall Score</div>'
        f'<div class="health-score-value">{score}<small>/100</small></div>'
        f'<p>{caveat}</p></div>'
        f'<div class="health-signals">{signal_rows}</div></div>',
        unsafe_allow_html=True,
    )


def render_skipped_files():
    """Show a persistent, inspectable record of repository exclusions."""
    skipped = st.session_state.skipped_files or []
    if not skipped:
        return
    with st.expander(f"Skipped files · {len(skipped)} 个项目未纳入分析", expanded=False):
        display = pd.DataFrame(skipped).rename(columns={
            "path": "文件 / 目录", "reason": "跳过原因", "size_mb": "大小 (MB)"
        })
        st.dataframe(display, width="stretch", hide_index=True)


def render_section_card(icon, title, description):
    st.markdown(
        f'<div class="cc-card"><div class="cc-icon">{html.escape(str(icon))}</div>'
        f'<h4>{html.escape(str(title))}</h4><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_page_heading(eyebrow, title, description):
    """Render the shared product-page hierarchy without changing page behavior."""
    st.markdown(
        '<div class="page-heading"><div class="page-heading-copy">'
        f'<div class="section-label">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(description)}</p>'
        '</div><div class="page-context"><span></span>STATIC AUDIT WORKSPACE</div></div>',
        unsafe_allow_html=True,
    )


def render_repair_flow():
    steps = (
        ("01", "Detected Issue"),
        ("02", "AI Solution"),
        ("03", "Generated Code"),
        ("04", "Download Patch"),
    )
    cards = "".join(
        '<div class="repair-step">'
        f'<div class="repair-index">{index}</div>'
        f'<div class="repair-label">{label}</div>'
        '</div>'
        for index, label in steps
    )
    st.markdown(f'<div class="repair-flow">{cards}</div>', unsafe_allow_html=True)


def split_agent_report(report):
    """Map the Agent's numbered Markdown sections to compact UI panels."""
    sections = {}
    current = None
    for line in (report or "").splitlines():
        match = re.match(r"^##\s+([1-6])\.\s*(.+)$", line.strip())
        if match:
            current = int(match.group(1))
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {index: "\n".join(lines).strip() for index, lines in sections.items()
    }     

NAVIGATION = [
    "Home", "Code Analysis", "Data Quality", "AI Review", "Repair", "Report",
]

NAV_LABELS = {
    "Home": "Overview",
    "Code Analysis": "Code Review",
    "Data Quality": "Data Quality",
    "AI Review": "AI Review",
    "Repair": "Repair",
    "Report": "Report",
}


def navigate_to(page, import_mode=None):
    """Navigation callback used by product entry buttons."""
    st.session_state.active_page = page
    if import_mode:
        st.session_state.import_mode = import_mode


def render_sidebar_navigation():
    """Render a compact workspace sidebar without radio-style navigation."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="sidebar-mark">CD</div>'
            '<div><strong>CodeData-Copilot</strong><span>Project Auditor</span></div></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.analyzed_files:
            overview = st.session_state.project_overview or {}
            score = (st.session_state.health or {}).get("score", 0)
            st.markdown(
                '<div class="sidebar-project">'
                '<span class="sidebar-label">CURRENT PROJECT</span>'
                f'<strong>{html.escape(overview.get("project_name", "Project"))}</strong>'
                f'<span><i></i> Ready · {score}/100</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sidebar-project"><span class="sidebar-label">CURRENT PROJECT</span>'
                '<strong>No project loaded</strong><span>Waiting for import</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="sidebar-section-label">WORKSPACE</div>', unsafe_allow_html=True)
        for item in NAVIGATION:
            st.button(
                NAV_LABELS[item],
                key=f"nav_{item}",
                type="primary" if st.session_state.active_page == item else "secondary",
                width="stretch",
                on_click=navigate_to,
                args=(item,),
            )
        st.markdown('<div class="sidebar-foot"><span></span> LOCAL WORKSPACE<br>'
                    '<small>Static analysis · no source execution</small></div>', unsafe_allow_html=True)
    return st.session_state.active_page


def render_health_breakdown():
    health = st.session_state.health
    if not health:
        return
    score_col, progress_col = st.columns([1, 3])
    score_col.metric("Project Health", f"{health['score']}/100")
    with progress_col:
        st.caption("静态质量健康度")
        st.progress(health["score"] / 100)
    columns = st.columns(3)
    for column, (label, points) in zip(columns, health["breakdown"].items()):
        maximum = 30 if label == "项目完整性" else 35
        column.metric(label, f"{points}/{maximum}")


def render_overview_details():
    overview = st.session_state.project_overview or {}
    if not overview:
        return
    stats = st.session_state.project_stats or {}
    last_analyzed = st.session_state.last_analyzed_at or "This session"
    st.markdown(
        '<div class="project-identity">'
        '<div><span class="panel-kicker">CURRENT PROJECT</span>'
        f'<h3>{html.escape(overview.get("project_name", "Project"))}</h3>'
        f'<p>{html.escape(overview.get("project_type", "Data Science Project"))} · '
        f'{stats.get("files", 0)} files · {stats.get("data_files", 0)} datasets</p></div>'
        '<div class="project-identity-status"><span><i></i> READY</span>'
        f'<small>Last analyzed · {html.escape(last_analyzed)}</small></div></div>',
        unsafe_allow_html=True,
    )
    render_project_health_panel()
    render_summary()
    st.markdown(
        '<div class="plain-section"><div class="panel-kicker">Project summary</div>'
        f'<p>{html.escape(overview.get("summary", "暂无项目摘要。"))}</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.subheader("使用语言")
        chips = "".join(f'<span class="tech-chip">{html.escape(item)}</span>'
                        for item in overview.get("languages", []))
        st.markdown(chips or "未识别", unsafe_allow_html=True)
    with right:
        st.subheader("主要库")
        chips = "".join(f'<span class="tech-chip">{html.escape(item)}</span>'
                        for item in overview.get("main_libraries", []))
        st.markdown(chips or "未识别到常见数据科学库", unsafe_allow_html=True)

    st.subheader("项目结构")
    structure = overview.get("project_structure", [])
    structure_rows = "".join(
        f'<li><span>{index:02d}</span><strong>{html.escape(item)}</strong>'
        '<small>Static structure signal</small></li>'
        for index, item in enumerate(structure, 1)
    )
    st.markdown(f'<ul class="structure-list">{structure_rows}</ul>', unsafe_allow_html=True)

    if overview.get("data_files"):
        st.subheader("数据文件信息")
        table = pd.DataFrame(overview["data_files"]).rename(columns={
            "file": "文件", "rows": "行数", "columns": "列数",
            "missing_columns": "含缺失值字段", "duplicate_rows": "重复行",
        })
        st.dataframe(table, width="stretch", hide_index=True)


def render_project_input():
    """Render the two existing import routes as a compact workstation control."""
    st.markdown(
        '<div class="section-heading"><div><span>PROJECT INPUT</span>'
        '<h2>Open an audit workspace</h2></div>'
        '<p>Files stay in the local analysis session.</p></div>',
        unsafe_allow_html=True,
    )
    import_mode = st.segmented_control(
        "Import method",
        ["Local Project", "GitHub Repository"],
        key="import_mode",
        label_visibility="collapsed",
        width="stretch",
    )
    if import_mode == "Local Project":
        st.caption("支持 .py、.ipynb、.csv；一次可选择多个文件，单文件最大 25 MB。")
        uploaded_files = st.file_uploader(
            "Drop project files here",
            type=["py", "ipynb", "csv"],
            accept_multiple_files=True,
            key="project_upload",
        )
        csv_uploads = [item for item in uploaded_files if item.name.lower().endswith(".csv")]
        if csv_uploads:
            try:
                preview_file = csv_uploads[0]
                preview = read_csv_preview(preview_file)
                st.markdown(f"#### Data preview · `{preview_file.name}`")
                st.dataframe(preview.head(10), width="stretch")
                st.caption(f"{preview.shape[0]} rows · {preview.shape[1]} columns")
            except Exception as exc:
                st.warning(f"CSV 预览失败：{exc}")
        option_col, action_col = st.columns([2.2, 1])
        with option_col:
            local_ai = st.checkbox(
                "Generate Qwen AI Review",
                value=True,
                key="local_ai",
                help="未配置 DASHSCOPE_API_KEY 时基础分析仍可完成。",
            )
        with action_col:
            analyze_local = st.button(
                "Analyze Project", type="primary", disabled=not uploaded_files, width="stretch"
            )
        if analyze_local:
            try:
                run_with_analysis_status(
                    lambda progress: run_uploaded_project(uploaded_files, local_ai, progress)
                )
                st.success("分析完成。")
            except Exception as exc:
                st.error(f"分析失败：{exc}")
    else:
        input_col, action_col = st.columns([4, 1])
        with input_col:
            github_url = st.text_input(
                "GitHub repository",
                placeholder="github.com/owner/repository",
                label_visibility="collapsed",
            )
        with action_col:
            analyze_github = st.button(
                "Analyze", type="primary", disabled=not github_url.strip(), width="stretch"
            )
        github_ai = st.checkbox("Generate Qwen AI Review", value=True, key="github_ai")
        st.caption(
            "仅分析 .py、.ipynb、.csv、.md；最多 100 个文件。导入内容在临时目录中处理并自动清理。"
        )
        if analyze_github:
            try:
                def import_repository(progress):
                    with load_github_project(github_url) as files:
                        analyze_project_files(
                            files,
                            github_ai,
                            {"source": "GitHub", "repository_url": github_url.strip()},
                            progress,
                        )
                run_with_analysis_status(import_repository)
                st.success("仓库分析完成。")
            except Exception as exc:
                st.error(str(exc))
                skipped = getattr(exc, "skipped_files", None)
                if skipped:
                    st.session_state.skipped_files = skipped
                    render_skipped_files()

    if st.session_state.analyzed_files:
        if st.session_state.analysis_error:
            st.warning(f"基础分析已完成，但 AI 报告生成失败：{st.session_state.analysis_error}")
        for warning in st.session_state.analysis_warnings:
            st.warning(warning)
        render_skipped_files()


init_state()
if st.session_state.active_page not in NAVIGATION:
    st.session_state.active_page = "Home"
if st.session_state.import_mode not in {"Local Project", "GitHub Repository"}:
    st.session_state.import_mode = "Local Project"

page = render_sidebar_navigation()


if page == "Home":
    status_label = "Ready" if st.session_state.analyzed_files else "Waiting for project"
    status_detail = st.session_state.last_analyzed_at or "No analysis in this session"
    st.markdown(
        f"""
        <div class="workspace-head">
          <div class="workspace-copy">
            <div class="workspace-meta"><span></span>OVERVIEW · AI AUDIT WORKSPACE</div>
            <h1>CodeData-Copilot</h1>
            <h2>AI-Powered Data Science Project Auditor</h2>
            <p>Project quality analysis, review and repair.</p>
          </div>
          <div class="brand-visual" aria-hidden="true">
            <div class="brand-visual-meta">
              <span>PROJECT GRAPH</span>
              <em><i></i>{html.escape(status_label)}</em>
            </div>
            <svg viewBox="0 0 360 92" role="img" aria-label="Abstract project data flow">
              <path class="graph-line" d="M31 46H90L116 20H180L208 47H274L302 24H338" />
              <path class="graph-line graph-line-teal" d="M90 46L119 72H204L228 47M274 47L300 72H338" />
              <path class="graph-dash" d="M45 21H87M150 47H180M238 19H278" />
              <rect class="graph-file" x="13" y="29" width="34" height="34" rx="4" />
              <path class="graph-file-mark" d="M22 40H38M22 46H34M22 52H37" />
              <rect class="graph-node" x="108" y="12" width="16" height="16" rx="3" />
              <rect class="graph-node graph-node-teal" x="196" y="39" width="16" height="16" rx="3" />
              <circle class="graph-point" cx="90" cy="46" r="4" />
              <circle class="graph-point graph-point-teal" cx="119" cy="72" r="4" />
              <circle class="graph-point" cx="180" cy="20" r="4" />
              <circle class="graph-point graph-point-teal" cx="274" cy="47" r="4" />
              <rect class="graph-terminal" x="330" y="16" width="16" height="16" rx="3" />
              <rect class="graph-terminal graph-terminal-teal" x="330" y="64" width="16" height="16" rx="3" />
            </svg>
            <small>{html.escape(status_detail)}</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_project_input()
    if st.session_state.analyzed_files:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_overview_details()
        with st.expander("评分依据", expanded=False):
            for explanation in (st.session_state.health or {}).get("explanation", []):
                st.write("• " + explanation)


elif page == "Data Quality":
    render_page_heading(
        "Data quality",
        "Dataset quality review",
        "检查完整性、非有限值、统计极端值、类型一致性、类别质量与特征冗余。",
    )
    if require_results():
        if not st.session_state.data_results:
            st.info("此项目没有可分析的 CSV 文件。")
            st.stop()
        choices = {entry["file"]: entry["result"] for entry in st.session_state.data_results}
        result = choices[st.selectbox("选择 CSV 文件", list(choices))]

        shape = result.get("shape", {})
        column_count = shape.get("columns", 0)
        missing_values = result.get("missing_values", {})
        overall_missing_rate = (
            sum(missing_values.values()) / column_count if column_count else 0.0
        )
        data_issues = result.get("issues") or build_data_issues(result)
        outlier_count = sum(item.get("count") or 0 for item in result.get("outliers", []))
        quality_score = issue_quality_score(data_issues)
        metric_rows, metric_columns, metric_missing, metric_duplicates, metric_outliers, metric_quality = st.columns(6)
        metric_rows.metric("Rows", f"{shape.get('rows', 0):,}", help="CSV 数据记录数")
        metric_columns.metric("Columns", column_count, help="数据字段数量")
        metric_missing.metric("Missing", f"{overall_missing_rate:.2%}",
                              help="所有字段缺失率的整体加权结果")
        metric_duplicates.metric("Duplicates", f"{result.get('duplicate_rows', 0):,}")
        metric_outliers.metric("Outliers", f"{outlier_count:,}", help="统计方法标记的潜在极端值")
        metric_quality.metric("Quality Score", f"{quality_score}/100")
        st.write("")

        (
            tab_issues, tab_missing, tab_outliers, tab_invalid, tab_types,
            tab_categories, tab_features, tab_correlation, tab_statistics,
        ) = st.tabs([
            "Quality Issues", "Missing Values", "Outliers", "Invalid Values",
            "Type Consistency", "Categories", "Feature Quality", "Correlation", "Statistics",
        ])

        with tab_issues:
            combined_issues = list(data_issues)
            combined_issues.extend(st.session_state.data_consistency_issues or [])
            if combined_issues:
                st.dataframe(issue_table(combined_issues), width="stretch", hide_index=True)
                st.caption("Potential / Suspicious 结论需要业务或统计复核；系统不会自动修改原始数据。")
            else:
                st.success("当前规则下未发现明显数据质量问题。")
        with tab_missing:
            missing = result.get("missing_values", {})
            if missing:
                missing_df = pd.DataFrame({
                    "字段": list(missing.keys()),
                    "缺失率": list(missing.values()),
                }).sort_values("缺失率", ascending=False)
                st.subheader("Missing values by column")
                st.bar_chart(missing_df.set_index("字段"), color="#39B8A5")
                display_df = missing_df.copy()
                display_df["缺失率"] = display_df["缺失率"].map(lambda x: f"{x:.2%}")
                st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.success("未发现缺失值。")
            token_issues = [
                item for item in result.get("missing_issues", [])
                if item.get("rule_id") != "DQ_MISSING_VALUES"
            ]
            if token_issues:
                st.subheader("Blank strings & suspected tokens")
                st.dataframe(issue_table(token_issues), width="stretch", hide_index=True)
                st.caption("Suspected Missing Token 不是已确认缺失值，需先确认业务语义。")

        with tab_outliers:
            outliers = result.get("outliers", [])
            if outliers:
                st.dataframe(issue_table(outliers), width="stretch", hide_index=True)
                st.caption("IQR/MAD 极端值是统计性 Potential，不等同于非法值。")
            else:
                st.success("在样本量和稳健阈值允许的字段中未发现明显统计极端值。")

        with tab_invalid:
            invalid = result.get("invalid_values", [])
            if invalid:
                st.dataframe(issue_table(invalid), width="stretch", hide_index=True)
            else:
                st.success("未发现非有限数值或显式约束违规。")
            st.caption("没有业务 constraints 时，系统不会仅凭字段名把数值范围判定为非法。")

        with tab_types:
            type_df = pd.DataFrame(result.get("data_types", {}).items(), columns=["字段", "数据类型"])
            st.dataframe(type_df, width="stretch", hide_index=True)
            if result.get("type_issues"):
                st.subheader("Parsing & normalization findings")
                st.dataframe(issue_table(result["type_issues"]), width="stretch", hide_index=True)

        with tab_categories:
            categories = result.get("categorical_issues", [])
            if categories:
                st.dataframe(issue_table(categories), width="stretch", hide_index=True)
            else:
                st.success("未发现明显的类别编码、稀有类别或疑似目标不平衡问题。")

        with tab_features:
            c1, c2, c3 = st.columns(3)
            c1.metric("Numeric", len(result.get("numeric_columns", [])))
            c2.metric("Categorical", len(result.get("categorical_columns", [])))
            c3.metric("Total", column_count)
            feature_issues = list(result.get("feature_issues", []))
            feature_issues.extend([
                item for item in data_issues
                if item.get("rule_id") in {"DQ_HIGH_CARDINALITY", "DQ_POTENTIAL_ID"}
                and item not in feature_issues
            ])
            if feature_issues:
                st.dataframe(issue_table(feature_issues), width="stretch", hide_index=True)
            st.caption("Constant、Potential ID 与高基数提示只用于审阅，不会自动删除字段。")

        with tab_correlation:
            correlation = result.get("correlation", {})
            if correlation:
                corr_df = pd.DataFrame(correlation)
                st.dataframe(corr_df, width="stretch")
                st.caption("相关系数接近 1 或 -1 表示线性相关程度较高，不代表因果关系。")
            else:
                st.info("当前数据中没有足够的数值字段用于相关性分析。")
            pairs = result.get("high_correlation_pairs", [])
            if pairs:
                st.subheader("High Correlation Pairs")
                st.dataframe(pd.DataFrame(pairs), width="stretch", hide_index=True)
                st.caption("高度相关表示潜在冗余，不代表必须删除其中一个特征。")
            for notice in result.get("analysis_limits", []):
                st.info(notice)

        with tab_statistics:
            statistics = result.get("numeric_statistics", {})
            if statistics:
                stats_df = pd.DataFrame.from_dict(statistics, orient="index")
                stats_df.index.name = "Column"
                st.dataframe(stats_df.reset_index(), width="stretch", hide_index=True)
            else:
                st.info("当前数据中没有数值字段统计。")


elif page == "Code Analysis":
    render_page_heading(
        "Code review",
        "Static findings",
        "按严重性、文件和行号审阅 AST 静态证据；展开条目查看原因与建议。",
    )
    if require_results():
        result = st.session_state.code_result or []
        flow = st.session_state.data_flow or {}
        with st.expander("Data Flow Summary", expanded=False):
            st.code(flow_summary_text(flow), language=None)
            flow_left, flow_middle, flow_right = st.columns(3)
            flow_left.metric(
                "Split",
                "Detected" if flow.get("split", {}).get("detected") else "Unable to verify",
            )
            flow_middle.metric(
                "Task",
                flow.get("task_type", {}).get("type", "Unknown"),
            )
            flow_right.metric(
                "Metrics",
                ", ".join(item.get("label", "") for item in flow.get("metrics", [])) or "Unable to verify",
            )
            features = flow.get("model_features", [])
            st.caption(
                "Model features: " + (", ".join(features) if features else "Unable to verify automatically.")
            )
            roles = flow.get("split_role_analysis", {})
            st.caption(
                "Roles: "
                f"train={roles.get('train_detected', False)} · "
                f"validation={roles.get('validation_detected', False)} · "
                f"test={roles.get('test_detected', False)}"
            )
            fairness = flow.get("fairness", {})
            if fairness.get("evaluation_exists"):
                st.caption(
                    "Fairness: " + fairness.get("coverage", "Evaluation detected")
                    + f" · before/after={fairness.get('before_after_comparison', False)}"
                )
        artifact_analysis = st.session_state.artifact_analysis or {}
        if artifact_analysis.get("groups"):
            with st.expander("Duplicate / Near-Duplicate Analysis Artifacts", expanded=False):
                for group in artifact_analysis["groups"]:
                    st.markdown(
                        f"**{group.get('type')}** · Primary: `{group.get('primary')}` · "
                        f"Similarity: {group.get('similarity', 1):.2%}"
                    )
                    st.caption("Affected files: " + ", ".join(group.get("files", [])))
                    for duplicate in group.get("files", [])[1:]:
                        st.caption(f"{duplicate} · Duplicate of {group.get('primary')}")
        issue_count = summarize_results(result, st.session_state.data_result)["code_issue_count"]
        score = (st.session_state.health or {}).get("breakdown", {}).get("代码质量", 0)
        quality_score = round((score / 35) * 100) if score else 0
        affected_files = len({location.get("file", "Python 文件") for location in result})
        metric_left, metric_middle, metric_right = st.columns([1.3, 1, 1])
        metric_left.metric("Code Quality Score", f"{quality_score}/100")
        metric_middle.metric("Detected Risks", issue_count)
        metric_right.metric("Affected Files", affected_files)
        st.progress(quality_score / 100)
        st.caption("Code Quality Score 来自现有静态评分的 100 分制展示，不代表代码运行结果。")

        if not result:
            st.success("当前规则下未发现明显代码问题。")
        else:
            repair_items = build_repair_suggestions(
                result, st.session_state.data_flow, st.session_state.data_results
            )
            st.write("")
            flat_issues = []
            for location in result:
                label = location.get("file", "Python 文件")
                if "cell" in location:
                    label += f" · Notebook Cell {location['cell']}"
                for issue in location.get("issues", []):
                    flat_issues.append({**issue, "file": label})
            st.markdown('<div class="section-label">ISSUE INDEX</div>', unsafe_allow_html=True)
            issue_index_df = issue_table(flat_issues, include_file=True)
            issue_index_df = issue_index_df[
                ["Severity", "Confidence", "Category", "Issue", "Affected Files", "Line"]
            ]
            st.dataframe(
                issue_index_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Severity": st.column_config.TextColumn(width="small"),
                    "Confidence": st.column_config.TextColumn(width="small"),
                    "Category": st.column_config.TextColumn(width="medium"),
                    "Issue": st.column_config.TextColumn(width="large"),
                    "Affected Files": st.column_config.TextColumn(width="large"),
                    "Line": st.column_config.TextColumn(width="small"),
                },
            )
            st.markdown('<div class="section-label issue-detail-label">FINDING DETAILS</div>', unsafe_allow_html=True)
            issue_index = 0
            for location in result:
                label = location.get("file", "Python 文件")
                if "cell" in location:
                    label += f" · Notebook Cell {location['cell']}"
                for issue in location.get("issues", []):
                    issue_index += 1
                    issue_type = issue.get("type", "代码问题")
                    severity = str(issue.get("severity", "Low")).upper()
                    confidence = issue.get("confidence", "Medium")
                    category = issue.get("category", "Code Quality")
                    repair = next(
                        (item for item in repair_items if item.get("problem") == issue_type),
                        None,
                    )
                    reason = issue.get("message", "暂无说明")
                    evidence = issue.get("evidence") or issue.get("source_snippet") or reason
                    issue_location = label
                    if issue.get("line"):
                        issue_location += f" · Line {issue['line']}"
                    suggestion = issue.get("recommendation") or (
                        repair.get("explanation") if repair else "结合项目上下文复核，并在测试环境验证修改。"
                    )
                    affected = issue.get("affected_files") or location.get("affected_files") or [location.get("file", "Python 文件")]
                    with st.expander(
                        f"{issue_index:02d}  ·  {severity}  ·  {confidence} confidence  ·  {issue_type}  ·  {issue_location}"
                    ):
                        st.markdown(
                            '<div class="finding-detail">'
                            f'<span class="risk-badge severity-{severity.lower()}">{severity}</span>'
                            f'<code>{html.escape(issue_location)}</code>'
                            f'<dl><dt>Category</dt><dd>{html.escape(category)} · {html.escape(str(confidence))} confidence</dd>'
                            f'<dt>Evidence</dt><dd>{html.escape(str(evidence))}</dd>'
                            f'<dt>Affected Files</dt><dd>{html.escape(", ".join(affected))}</dd>'
                            f'<dt>Why it matters</dt><dd>{html.escape(reason)}</dd>'
                            f'<dt>Recommendation</dt><dd>{html.escape(suggestion)}</dd></dl></div>',
                            unsafe_allow_html=True,
                        )
                        if issue.get("source_snippet") or (repair and repair.get("after_code")):
                            if issue.get("source_snippet"):
                                st.caption("SOURCE EVIDENCE")
                                st.code(issue["source_snippet"], language="python")
                            if repair and repair.get("after_code"):
                                st.caption("SUGGESTED FIX")
                                st.code(repair["after_code"], language="python")


elif page in ("AI Review", "Repair"):
    if page == "AI Review":
        render_page_heading(
            "AI review",
            "Intelligence review",
            "将静态证据、数据质量结果与本地知识库组织为可审阅的专业结论。",
        )
    else:
        render_page_heading(
            "Repair",
            "Review fixes before they become patches",
            "从检测问题到补丁导出保持完整审阅链路，系统不会直接修改项目文件。",
        )
    if require_results():
        if page == "Repair":
            suggestions = st.session_state.repair_suggestions or build_repair_suggestions(
                st.session_state.code_result,
                st.session_state.data_flow,
                st.session_state.data_results,
            )
            patch_text = st.session_state.patch_text or build_unified_patch(suggestions)
            render_repair_flow()
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Repair suggestions", len(suggestions))
            metric_b.metric(
                "Validation passed",
                sum(item.get("validation_status") in {"Fully Validated", "Syntax Valid"} for item in suggestions),
            )
            metric_c.metric("Project modified", "No")

            st.markdown(
                '<div class="patch-note">系统只生成建议和补丁文件，不会直接修改导入的用户项目。'
                '请审阅修改前后代码，并在确认后下载。</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            if suggestions:
                for item in suggestions:
                    with st.expander(f"Detected Issue · {item['problem']}", expanded=True):
                        st.markdown(
                            '<div class="repair-record-head">'
                            f'<div><span>FILE</span><code>{html.escape(item["location"])}</code></div>'
                            f'<div><span>ISSUE</span><strong>{html.escape(item["problem"])}</strong></div>'
                            '<div><span>REPAIR STATUS</span><em>Review required</em></div>'
                            '</div>'
                            f'<p class="repair-explanation">{html.escape(item["explanation"])}</p>',
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"REPAIR VALIDATION · {item.get('validation_status', 'Not Automatically Verifiable')}"
                        )
                        if item.get("validation_detail"):
                            st.write(item["validation_detail"])
                        before, after = st.columns(2)
                        with before:
                            st.caption("BEFORE")
                            if item["before_code"]:
                                st.code(item["before_code"], language="python")
                            else:
                                st.info("未提供原始代码片段，无法生成可应用的 diff。")
                        with after:
                            st.caption("AFTER")
                            st.code(item["after_code"], language="python")
            else:
                st.success("当前静态规则下没有需要生成的修复建议。")

            st.subheader("Patch export")
            if patch_text:
                with st.expander("预览 unified diff", expanded=True):
                    st.code(patch_text, language="diff")
                confirmed = st.checkbox(
                    "我已审阅修改前代码、修改后代码和修改说明，确认导出 Patch。",
                    key="patch_confirmed",
                )
                st.download_button(
                    "Download .patch",
                    data=patch_text.encode("utf-8"),
                    file_name="fix_issue.patch",
                    mime="text/x-diff",
                    type="primary",
                    disabled=not confirmed,
                )
            elif suggestions:
                st.warning("检测到了修复建议，但缺少真实原始代码片段，因此没有生成可能误导的 Patch。")

            if st.session_state.agent_report:
                with st.expander("AI Review 中的补充修复说明"):
                    st.markdown(st.session_state.agent_report)
        else:
            suggestions = st.session_state.repair_suggestions or build_repair_suggestions(
                st.session_state.code_result,
                st.session_state.data_flow,
                st.session_state.data_results,
            )
            reports = [item["result"] for item in st.session_state.data_results]
            data_issues = [issue for report in reports for issue in build_data_issues(report)]
            summary_left, summary_middle, summary_right = st.columns(3)
            summary_left.metric("Detected problems", len(suggestions) + len(data_issues))
            summary_middle.metric("Knowledge coverage", f"{len(load_knowledge_documents())} topics")
            summary_right.metric("Qwen review", "Ready" if st.session_state.agent_report else "Not generated")

            if suggestions or data_issues:
                with st.expander("Static evidence", expanded=False):
                    for item in suggestions:
                        st.markdown(f"**Code · {item['location']} · {item['problem']}**")
                        st.write(item["reason"])
                    for issue in data_issues:
                        st.markdown(f"**Data · {issue['problem']}**")
                        st.write(issue["detail"])

            if st.session_state.agent_report:
                sections = split_agent_report(st.session_state.agent_report)
                panel_map = (
                    ("Summary", sections.get(1, "暂无项目总结。")),
                    ("Key Findings", "\n\n".join(filter(None, (sections.get(2), sections.get(3))))),
                    ("Recommendations", sections.get(5, "暂无优化建议。")),
                    ("Priority Actions", "\n\n".join(filter(None, (sections.get(4), sections.get(6))))),
                )
                for index, (title, content) in enumerate(panel_map, 1):
                    st.markdown(
                        '<div class="review-section-head">'
                        f'<span>{index:02d}</span><div><small>AI REVIEW</small>'
                        f'<h3>{html.escape(title)}</h3></div></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(content or "暂无内容。")
            elif st.session_state.analysis_error:
                st.warning(f"AI Review 未生成：{st.session_state.analysis_error}")
            else:
                st.info("本次分析尚未调用 Qwen Agent。静态诊断仍然有效，可在下方生成 Review。")

            if st.button("Generate / refresh AI Review", type="primary"):
                try:
                    with st.status("正在生成知识增强 Review", expanded=True) as review_status:
                        knowledge_context = retrieve_relevant_knowledge(
                            st.session_state.code_result,
                            st.session_state.data_results,
                            st.session_state.project_context,
                        )
                        st.write("✓ 已选择相关质量准则")
                        st.session_state.agent_report = run_agent(
                            st.session_state.code_result,
                            st.session_state.data_results,
                            st.session_state.project_context,
                            knowledge_context=knowledge_context,
                        )
                        st.write("✓ Qwen Review 已生成")
                        rebuild_session_report()
                        st.write("✓ 报告已同步更新")
                        review_status.update(label="AI Review 已更新", state="complete", expanded=False)
                    st.session_state.analysis_error = None
                    st.rerun()
                except Exception as exc:
                    st.session_state.analysis_error = str(exc)
                    st.error(f"Agent 调用失败：{exc}")

        st.divider()
        st.caption(
            "AI 建议用于辅助审查。涉及数据删除、特征处理和模型修改时，"
            "请结合业务背景复核后再执行。"
        )


elif page == "Report":
    render_page_heading(
        "Report",
        "Analysis report",
        "预览并导出同源的 Markdown 与 PDF 审查报告。",
    )
    if require_results():
        if not st.session_state.report_data:
            rebuild_session_report()
        report = st.session_state.report_data or {}
        overview = report.get("project_overview", {})
        safe_project_name = re.sub(
            r"[^A-Za-z0-9._-]+", "-", overview.get("project_name", "project")
        ).strip("-") or "project"
        generated_time = st.session_state.last_analyzed_at or "This session"
        st.markdown(
            '<div class="report-status-bar">'
            f'<div><span>PROJECT</span><strong>{html.escape(overview.get("project_name", "Project"))}</strong></div>'
            f'<div><span>GENERATED</span><strong>{html.escape(generated_time)}</strong></div>'
            '<div><span>REPORT STATUS</span><strong class="status-ready"><i></i> Ready</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label report-actions-label">EXPORT</div>', unsafe_allow_html=True)
        col_markdown, col_pdf, col_space = st.columns([1, 1, 2])
        with col_markdown:
            st.download_button(
                "Download Markdown",
                data=(st.session_state.report_markdown or "").encode("utf-8"),
                file_name=f"{safe_project_name}-analysis-report.md",
                mime="text/markdown; charset=utf-8",
                type="primary",
                width="stretch",
            )
        with col_pdf:
            if st.session_state.report_pdf:
                st.download_button(
                    "Download PDF",
                    data=st.session_state.report_pdf,
                    file_name=f"{safe_project_name}-analysis-report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.button("Download PDF", disabled=True, width="stretch")
        if not st.session_state.report_pdf:
            st.warning(
                "PDF 暂未生成。Markdown 报告仍可正常下载。"
                + (f" 原因：{st.session_state.report_error}" if st.session_state.report_error else "")
            )
        with st.expander("Preview", expanded=False):
            st.markdown(st.session_state.report_markdown or "暂无报告内容。")
