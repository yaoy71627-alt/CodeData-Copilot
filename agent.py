import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from copy import deepcopy
import time
import hashlib
from qwen_runtime import (
    load_settings, classify_error, InvalidAIResponse, JobBudgetExceeded,
    make_prompt, plan_batches, parse_payload, completion_content,
    SYSTEM_PROMPT, retryable_error,
    compact_knowledge,
)
from consistency_validator import build_analysis_facts, build_findings, group_findings, render_structured_review


# ======================
# 加载环境变量
# ======================

load_dotenv()


KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
MAX_PROJECT_CONTEXT_CHARS = 16_000

KNOWLEDGE_TOPICS = {
    "data_quality.md": ("missing", "duplicate", "dtype", "outlier", "imbalance", "csv", "缺失", "重复", "异常值", "不平衡", "数据质量"),
    "code_quality.md": ("syntax", "code", "source_snippet", "代码", "依赖", "异常"),
    "data_leakage.md": ("leakage", "fit_transform", "train_test_split", "target", "泄漏", "泄露", "测试集", "目标"),
    "feature_engineering.md": ("feature", "encoder", "selector", "correlation", "scaler", "特征", "相关", "编码", "标准化"),
    "model_evaluation.md": ("metric", "score", "predict", "evaluation", "classification", "regression", "cross_val", "评估", "评价", "分类", "回归", "交叉验证"),
    "ml_best_practice.md": ("pipeline", "preprocess", "model", "机器学习", "预处理"),
    "reproducibility.md": ("seed", "random_state", "requirements", "version", "参数", "随机", "复现", "依赖"),
    "notebook_quality.md": ("notebook", "ipynb", "cell", "global", "path", "dropna", "读写", "执行顺序", "硬编码"),
    "sklearn_pandas_best_practice.md": ("sklearn", "pandas", "pipeline", "fit", "transform", "loc", "copy", "merge", "read_csv"),
    "outliers_invalid_values.md": ("outlier", "extreme", "iqr", "mad", "non-finite", "constraint", "mixed type", "异常值", "极端值", "非法值", "类型"),
    "security_reliability.md": ("secret", "credential", "eval", "exec", "shell", "exception", "path", "security", "凭据", "异常处理", "可靠性"),
    "data_science_code_review.md": ("leakage", "validation", "evaluation", "random_state", "cross-validation", "test set", "target leakage", "验证", "评价", "复现"),
}


class KnowledgeReadError(RuntimeError):
    """Knowledge storage is unavailable; safe to continue without it."""


class MissingAPIKeyError(ValueError):
    pass


def load_knowledge_documents():
    """Read a fresh snapshot so removal or repair is visible on the next run."""
    try:
        if not KNOWLEDGE_DIR.is_dir():
            raise KnowledgeReadError("知识库读取失败：目录不存在或不可访问。")
        documents = tuple(
            (path.name, path.read_text(encoding="utf-8"))
            for path in sorted(KNOWLEDGE_DIR.glob("*.md"))
        )
        if not documents:
            raise KnowledgeReadError("知识库读取失败：没有 Markdown 文件。")
        return documents
    except (OSError, UnicodeError) as exc:
        raise KnowledgeReadError("知识库读取失败：文件不可读或不是有效 UTF-8。") from exc


def safe_retrieve_knowledge(code_result, data_result, project_context=None):
    """Return context plus auditable status; never expose raw exception payloads."""
    try:
        context = retrieve_relevant_knowledge(code_result, data_result, project_context)
        sources = [line[4:].strip() for line in context.splitlines() if line.startswith("### ")]
        rules = [line[2:].strip() for line in context.splitlines() if line.startswith("- ")]
        return context, {
            "status": "success" if context else "empty",
            "reason": "知识准则检索成功。" if context else "知识检索完成，未返回相关准则。",
            "sources": sources, "rules": rules,
        }
    except Exception as exc:
        reading = isinstance(exc, (KnowledgeReadError, OSError, UnicodeError))
        return "", {
            "status": "read_failed" if reading else "retrieval_failed",
            "reason": ("知识库读取失败：请检查目录、文件权限及 UTF-8 编码。" if reading
                       else "知识库检索失败：准则筛选发生异常。") + " 已跳过知识增强。",
            "sources": [], "rules": [],
        }


def classify_ai_error(exc):
    """Use safe actionable messages, never SDK responses containing credentials."""
    if isinstance(exc, MissingAPIKeyError):
        return {"status": "missing_api_key", "reason": "API Key 缺失：未配置 DASHSCOPE_API_KEY。"}
    return classify_error(exc)


def ai_review_status(review, knowledge_context=""):
    from ui_review_helpers import normalized_ai
    status = normalized_ai(deepcopy(getattr(review, "ai_status", {})) or classify_error(InvalidAIResponse()), getattr(review, "coverage", None))
    status["validation_rejections"] = deepcopy(getattr(review, "coverage", {}).get("rejected_explanations", []))
    referenced = status.get("knowledge_reference", {}).get("used", bool(knowledge_context))
    status["used_knowledge"] = bool(referenced) and status["status"] in {"success", "partial_success"}
    return status


def load_knowledge_base():
    """Return the complete local knowledge base for compatibility and inspection."""
    return "\n\n".join(content for _, content in load_knowledge_documents())


def retrieve_relevant_knowledge(code_result, data_result, project_context=None, max_items=10):
    """Select a compact set of relevant rules without exposing storage filenames."""
    evidence = json.dumps(
        {"code": code_result or [], "data": data_result or [], "project": project_context or {}},
        ensure_ascii=False,
        default=str,
    ).lower()
    ranked = []
    documents = load_knowledge_documents()
    for name, content in documents:
        topic_keywords = KNOWLEDGE_TOPICS.get(name, ())
        topic_score = sum(1 for keyword in topic_keywords if keyword in evidence)
        title = next((line[2:].strip() for line in content.splitlines() if line.startswith("# ")),
                     Path(name).stem.replace("_", " ").title())
        for line in content.splitlines():
            fragment = line.strip()
            if not fragment.startswith("- "):
                continue
            fragment_lower = fragment.lower()
            fragment_score = sum(
                1 for keyword in topic_keywords
                if keyword in evidence and keyword in fragment_lower
            )
            score = topic_score * 2 + fragment_score
            if score:
                ranked.append((score, title, fragment))
    if not ranked:
        for name, content in documents:
            if name not in {"code_quality.md", "data_quality.md"}:
                continue
            title = next((line[2:].strip() for line in content.splitlines() if line.startswith("# ")),
                         "Quality Review")
            ranked.extend((1, title, line.strip()) for line in content.splitlines() if line.startswith("- "))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

    groups = {}
    seen = set()
    for _, title, fragment in ranked:
        if len(seen) >= max_items:
            break
        if fragment in seen:
            continue
        groups.setdefault(title, []).append(fragment)
        seen.add(fragment)
    return "\n\n".join(f"### {title}\n" + "\n".join(items) for title, items in groups.items())


def _compact_project_context(project_context):
    """Keep repository context useful without sending an unbounded prompt."""
    if not project_context:
        return "未提供额外项目上下文。"
    serialized = json.dumps(project_context, ensure_ascii=False, indent=2, default=str)
    if len(serialized) <= MAX_PROJECT_CONTEXT_CHARS:
        return serialized
    return serialized[:MAX_PROJECT_CONTEXT_CHARS] + "\n…（项目上下文已截断）"


def _repair_generation_context(data_flow, location, issue):
    candidates = list((data_flow or {}).get("files") or [data_flow or {}])
    flow = next(
        (item for item in candidates if item.get("file") == location.get("file")),
        {} if (data_flow or {}).get("files") else data_flow or {},
    )
    cell = location.get("cell")
    line = issue.get("line")
    units = [
        unit for unit in flow.get("code_units", [])
        if cell is None or unit.get("cell") == cell
    ]
    available = []
    for name, definitions in (flow.get("definitions") or {}).items():
        if any(
            (
                cell is None
                and definition.get("cell") is None
                and (line is None or (definition.get("line") or 0) < line)
            )
            or (
                cell is not None
                and definition.get("cell") is not None
                and (
                    definition.get("cell") < cell
                    or (
                        definition.get("cell") == cell
                        and (line is None or (definition.get("line") or 0) < line)
                    )
                )
            )
            for definition in definitions
        ):
            available.append(name)
    return {
        "file": location.get("file"),
        "cell": cell,
        "line": line,
        "surrounding_cell_code": units[0].get("code") if units else None,
        "upstream_definitions": sorted(available),
        "downstream_usages": [
            item for item in flow.get("lineage", [])
            if item.get("from") in available or item.get("to") in available
        ],
        "available_variables": sorted(available),
        "model_features": list(flow.get("model_features", [])),
        "target": flow.get("target"),
        "split_state": flow.get("split"),
        "existing_preprocessing": flow.get("preprocessing", []),
        "candidate_model_types": flow.get("possible_object_types", {}),
    }


def build_repair_suggestions(code_result, data_flow=None, data_results=None):
    """Offline examples for known rules; never edit the analyzed project."""
    examples = {
        "Data Leakage Risk": (
            "可能在划分数据前拟合变换器，导致测试集信息泄露。",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "scaler.fit(X_train)\nX_train = scaler.transform(X_train)\nX_test = scaler.transform(X_test)"
        ),
        "Potential Data Leakage": (
            "已检测到划分和变换，但静态检查不能确认顺序；请确保仅在训练集上拟合。",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "scaler.fit(X_train)\nX_train = scaler.transform(X_train)\nX_test = scaler.transform(X_test)"
        ),
        "Missing Value Handling Risk": (
            "直接删除缺失记录可能损失样本；先评估缺失机制和比例。",
            "imputer = SimpleImputer(strategy='median')\n"
            "X_train = imputer.fit_transform(X_train)\nX_test = imputer.transform(X_test)"
        ),
        "Test-set Used for Model Comparison": (
            "使用独立验证集选择候选模型，仅在方案锁定后使用测试集。",
            None,
        ),
        "Test-set Used for Hyperparameter Comparison": (
            "针对每个 estimator 分别定义合法参数空间，并在验证集或交叉验证中选择配置。",
            None,
        ),
        "Repeated Test-set Evaluation Requires Review": (
            "静态证据不足以生成安全 Patch；请先明确循环中的比较对象并保留独立最终测试集。",
            None,
        ),
    }
    suggestions = []
    flow = data_flow or {}
    model_features = list(flow.get("model_features", []))
    data_types = {}
    missing_values = {}
    for entry in data_results or []:
        result = entry.get("result", entry)
        data_types.update(result.get("data_types", {}))
        missing_values.update(result.get("missing_values", {}))
    for location in code_result or []:
        label = location.get("file", "代码文件")
        if "cell" in location:
            label += f" · Cell {location['cell']}"
        for issue in location.get("issues", []):
            kind = issue.get("type")
            if kind in examples:
                reason, corrected = examples[kind]
                original = issue.get("source_snippet")
                referenced_columns = []
                scope = None
                if kind == "Missing Value Handling Risk":
                    referenced_columns = [column for column in model_features if missing_values.get(column, 0) > 0]
                    corrected = None
                    scope = "model_input"
                    reason = (
                        "缺失值处理需协调样本筛选、训练/测试边界、填补器拟合与下游输入；"
                        "不能把 dropna 单行替换为全量数据中位数/众数填补来证明安全。"
                        "仅对实际模型字段提出核查方向，降级为 Recommendation Only。"
                    )
                elif kind in {"Data Leakage Risk", "Potential Data Leakage"}:
                    corrected = None
                    reason = (
                        "该问题需要同时调整划分、变换器拟合和后续变量引用；"
                        "单行替换无法保证下游兼容，因此降级为 Recommendation Only。"
                    )
                elif not original:
                    corrected = None
                    reason = "缺少可逐字定位的原始代码，仅保留 Recommendation Only。"
                generation_context = _repair_generation_context(data_flow, location, issue)
                suggestions.append({
                    "location": label,
                    "file": location.get("file"),
                    "cell": location.get("cell"),
                    "line": issue.get("line"),
                    "problem": kind,
                    "reason": reason,
                    "original_code": original,
                    "corrected_code": corrected,
                    "before_code": original,
                    "after_code": corrected,
                    "conceptual_code": None,
                    "requires_multi_location": kind in {"Data Leakage Risk", "Potential Data Leakage", "Missing Value Handling Risk"},
                    "explanation": reason,
                    "requires_confirmation": True,
                    "referenced_columns": referenced_columns,
                    "scope": scope,
                    "repair_context": generation_context,
                })
    return suggestions


def get_client():
    """按需创建千问客户端，避免未配置密钥时网页无法启动。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key or not api_key.strip():
        raise MissingAPIKeyError(
            "未检测到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env，"
            "并填写阿里云百炼 API Key。"
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("未安装 openai 依赖；请运行 python -m pip install -r requirements.txt。") from exc

    settings = load_settings()
    return OpenAI(
        api_key=api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=0,
    )



# ======================
# Agent核心
# ======================

def run_agent(
    code_result,
    data_result,
    project_context=None,
    knowledge_context=None,
    analysis_facts=None,
    previous_review=None,
):
    """
    数据科学项目Review Agent

    输入:
        code_result:
            代码分析结果

        data_result:
            数据质量分析结果

        project_context:
            文件清单、README 摘要和用户代码上下文（可选）

    输出:
        LLM生成的诊断报告
    """

    # None means not attempted; an explicit empty string means attempted/skipped.
    relevant_knowledge = knowledge_context
    if relevant_knowledge is None:
        relevant_knowledge, _ = safe_retrieve_knowledge(code_result, data_result, project_context)
    has_knowledge = bool(relevant_knowledge)
    relevant_knowledge = relevant_knowledge or "本次未使用知识库。仅依据静态证据审查，不得声称使用了本地准则。"
    entries = data_result or []
    if isinstance(entries, dict):
        entries = [{"result": entries}]
    facts = deepcopy(analysis_facts) if analysis_facts is not None else build_analysis_facts(
        code_result, entries, {},
    )
    facts["findings"] = build_findings(code_result, entries)
    facts["review_problems"] = group_findings(facts["findings"])
    settings = load_settings()
    findings = facts["findings"]
    fingerprint = hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    if not findings:
        review = render_structured_review({}, facts)
        review.ai_status = {"status": "not_needed", "reason": "没有正式 Finding，本次未调用模型。", "batches": []}
        return review
    if previous_review is not None:
        from consistency_validator import ensure_review_coverage
        previous_review = ensure_review_coverage(previous_review, facts)
    previous_status = getattr(previous_review, "ai_status", {})
    cached = {}
    if (previous_status.get("status") == "partial_success" and previous_status.get("facts_fingerprint") == fingerprint
            and previous_status.get("model") == settings.model and previous_status.get("region") == settings.region):
        cached = {fid: fields for fid, fields in getattr(previous_review, "coverage", {}).get("accepted_annotations", {}).items()
                  if fid in {f["id"] for f in findings} and all(fields.get(k) for k in ("cause", "potential_impact", "recommendation"))}
    pending = [f for f in findings if f["id"] not in cached]
    batches = plan_batches(pending, facts, relevant_knowledge, settings, project_context)
    client = get_client() if batches else None
    started, spent = time.monotonic(), 0
    payload = {"explanations": [{"finding_id": fid, **fields} for fid, fields in cached.items()], "unverified_suggestions": []}
    telemetry, errors, rejected = [], [], False
    stop_job = False
    for batch in batches[:settings.max_batches]:
        original_ids = [finding["id"] for finding in batch]
        remaining_batch = list(batch)
        record = {
            "finding_ids": original_ids,
            "attempts": [],
            "token_estimate_method": "conservative character heuristic; not actual usage",
        }
        terminal_error = None
        for attempt_index in range(settings.retry_count + 1):
            prompt = make_prompt(remaining_batch, facts, relevant_knowledge, project_context)
            estimate = len(prompt) + len(SYSTEM_PROMPT)
            remaining_time = settings.job_timeout - (time.monotonic() - started)
            available = settings.token_budget - spent - estimate
            if remaining_time <= 0 or available < 128 or estimate > settings.max_input_chars:
                # A retry budget guard must not hide the actual response failure
                # that caused the retry (for example a timeout or invalid JSON).
                # Preserve that root cause for the public status and telemetry;
                # only use budget_exceeded when no request failure preceded it.
                if terminal_error is None:
                    terminal_error = JobBudgetExceeded()
                errors.append(classify_ai_error(terminal_error))
                stop_job = True
                break
            output_limit = min(settings.max_output_tokens, available)
            attempt_record = {
                "attempt": attempt_index + 1,
                "input_characters": estimate,
                "user_prompt_characters": len(prompt),
                "estimated_input_tokens": estimate,
                "max_output_tokens": output_limit,
                "finding_ids": [finding["id"] for finding in remaining_batch],
            }
            charge = estimate + output_limit
            try:
                content, metadata = completion_content(
                    client, prompt, settings, remaining_time, output_limit,
                )
                attempt_record.update(metadata)
                charge = max(charge, metadata["usage"].get("total_tokens", 0))
                part, invalid_items = parse_payload(
                    content, {finding["id"] for finding in remaining_batch},
                )
                accepted_ids = {item["finding_id"] for item in part["explanations"]}
                payload["explanations"].extend(part["explanations"])
                payload["unverified_suggestions"].extend(part["unverified_suggestions"])
                remaining_batch = [
                    finding for finding in remaining_batch if finding["id"] not in accepted_ids
                ]
                attempt_record.update({
                    "status": "parsed" if not remaining_batch else "partial_response",
                    "accepted_count": len(accepted_ids),
                    "remaining_count": len(remaining_batch),
                    "invalid_items_rejected": bool(invalid_items),
                })
                record["attempts"].append(attempt_record)
                spent += charge
                if not remaining_batch:
                    terminal_error = None
                    break
                terminal_error = InvalidAIResponse()
            except Exception as exc:
                terminal_error = exc
                attempt_record.update(classify_ai_error(exc))
                attempt_record["remaining_count"] = len(remaining_batch)
                record["attempts"].append(attempt_record)
                spent += charge

            if (terminal_error is not None and retryable_error(terminal_error)
                    and attempt_index < settings.retry_count):
                delay = settings.retry_backoff * (2 ** attempt_index)
                if time.monotonic() + delay - started < settings.job_timeout:
                    time.sleep(delay)
                    continue
            if terminal_error is not None:
                errors.append(classify_ai_error(terminal_error))
                rejected = rejected or isinstance(terminal_error, InvalidAIResponse)
                if not isinstance(terminal_error, InvalidAIResponse):
                    stop_job = True
            break
        record["retry_count"] = max(0, len(record["attempts"]) - 1)
        record["status"] = (
            "parsed" if not remaining_batch
            else classify_ai_error(terminal_error or InvalidAIResponse())["status"]
        )
        record["response_success_count"] = len(original_ids) - len(remaining_batch)
        record["response_failure_count"] = len(remaining_batch)
        record["input_characters"] = sum(item.get("input_characters", 0) for item in record["attempts"])
        record["user_prompt_characters"] = sum(item.get("user_prompt_characters", 0) for item in record["attempts"])
        record["estimated_input_tokens"] = sum(item.get("estimated_input_tokens", 0) for item in record["attempts"])
        record["max_output_tokens"] = sum(item.get("max_output_tokens", 0) for item in record["attempts"])
        record["failure_reasons"] = list(dict.fromkeys(
            item.get("status") for item in record["attempts"]
            if item.get("status") not in {None, "parsed"}
        ))
        usage = {}
        for attempt in record["attempts"]:
            for key, value in (attempt.get("usage") or {}).items():
                usage[key] = usage.get(key, 0) + value
        record["usage"] = usage
        record["request_ids"] = [
            item["request_id"] for item in record["attempts"] if item.get("request_id")
        ]
        record["request_id"] = record["request_ids"][-1] if record["request_ids"] else None
        telemetry.append(record)
        if stop_job:
            break
    if len(batches) > settings.max_batches:
        errors.append(classify_ai_error(JobBudgetExceeded()))
    review = render_structured_review(payload if payload["explanations"] else None, facts)
    annotations = review.coverage.get("accepted_annotations", {})
    accepted = sum(all(fields.get(k) for k in ("cause", "potential_impact", "recommendation"))
                   for fields in annotations.values())
    for record in telemetry:
        accepted_ids = set(record["finding_ids"]) & set(annotations)
        record["success_count"] = len(accepted_ids)
        record["failure_count"] = len(record["finding_ids"]) - len(accepted_ids)
        if (record["failure_count"]
                and "semantic_validation_failed" not in record["failure_reasons"]):
            record["failure_reasons"].append("semantic_validation_failed")
    if accepted == len(findings) and not errors and not rejected:
        status, reason = "success", "Qwen Review 已生成。"
    elif accepted:
        status, reason = "partial_success", "仅部分 Finding 获得经校验的 AI 解释；其余已用结构化静态摘要补齐。"
    elif errors:
        # A pre-flight budget guard reached while retrying malformed output is
        # secondary evidence. Report the request/response failure that actually
        # caused the retry, while still retaining budget_exceeded in telemetry.
        error = next((item for item in errors if item["status"] not in {
            "invalid_ai_response", "budget_exceeded",
        }), None)
        if error is None:
            error = next((item for item in errors
                          if item["status"] == "invalid_ai_response"), errors[0])
        status, reason = error["status"], error["reason"]
    else:
        status, reason = "invalid_ai_response", "模型未返回有效的结构化解释；已保留静态诊断并确定性补齐。"
    if not accepted:
        review = type(review)("本次未生成有效 AI 解释；以下保留完整静态诊断与确定性摘要。\n\n" + review, review.coverage)
    review.ai_status = {"status": status, "reason": reason,
        "accepted_findings": accepted, "total_findings": len(findings),
        "batches": telemetry, "errors": errors, "model": settings.model,
        "region": settings.region, "key_region_checked": settings.key_region_checked,
        "budget_reserved_tokens": spent, "job_seconds": round(time.monotonic() - started, 3)}
    review.ai_status.update({"facts_fingerprint": fingerprint, "reused_findings": len(cached)})
    sent = compact_knowledge(relevant_knowledge)
    prior = previous_status.get("knowledge_reference", {}) if cached else {}
    review.ai_status["knowledge_reference"] = {
        "sources": list(dict.fromkeys(prior.get("sources", []) + [s[4:] for s in sent.splitlines() if s.startswith("### ")])),
        "rules": list(dict.fromkeys(prior.get("rules", []) + [s[2:] for s in sent.splitlines() if s.startswith("- ")])),
        "excerpt": sent != relevant_knowledge,
        "used": bool(has_knowledge and sent.strip() and telemetry) or bool(prior.get("used")),
    }
    if cached:
        review.ai_status["reason"] += " 已复用同一分析快照中校验通过的局部解释，仅请求未完成的问题。"
    return review



# ======================
# 本地测试
# ======================

if __name__ == "__main__":


    test_code_result = [

        {
            "cell":3,

            "issues":[

                {
                    "type":
                    "Data Leakage Risk",

                    "message":
                    "检测到fit_transform操作，可能在数据划分前使用全部数据拟合"
                }

            ]
        }

    ]


    test_data_result = {

        "shape":
        {
            "rows":1000,
            "columns":20
        },


        "missing_values":
        {
            "Age":0.2
        },


        "duplicate_rows":5

    }


    report = run_agent(
        test_code_result,
        test_data_result
    )


    print(report)
