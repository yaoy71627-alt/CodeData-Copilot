import os
import json
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


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


@lru_cache(maxsize=1)
def load_knowledge_documents():
    """Load all local Markdown knowledge documents as an immutable collection."""
    try:
        documents = []
        for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
            documents.append((path.name, path.read_text(encoding="utf-8")))
        if not documents:
            raise RuntimeError(f"知识库目录中没有 Markdown 文件：{KNOWLEDGE_DIR}")
        return tuple(documents)
    except OSError as exc:
        raise RuntimeError(f"无法读取机器学习知识库：{KNOWLEDGE_DIR}") from exc


@lru_cache(maxsize=1)
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
    for name, content in load_knowledge_documents():
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
        for name, content in load_knowledge_documents():
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
                if kind == "Missing Value Handling Risk" and model_features and original:
                    receiver_match = re.match(r"\s*([A-Za-z_]\w*)\.dropna\s*\(", original)
                    receiver = receiver_match.group(1) if receiver_match else None
                    referenced_columns = [
                        column for column in model_features if missing_values.get(column, 0) > 0
                    ]
                    if receiver and referenced_columns:
                        replacements = []
                        for column in referenced_columns:
                            dtype = str(data_types.get(column, ""))
                            if any(token in dtype for token in ("int", "float", "double", "decimal")):
                                value = f'{receiver}[{column!r}].median()'
                            else:
                                value = f'{receiver}[{column!r}].mode().iloc[0]'
                            replacements.append(f"{column!r}: {value}")
                        corrected = f"{receiver}.fillna({{{', '.join(replacements)}}})"
                        reason = "仅针对当前模型实际使用且存在缺失的字段，按数值/类别类型生成填补表达式。"
                        scope = "model_input"
                    else:
                        corrected = None
                        reason = "无法在当前执行位置确认可安全填补的模型字段，仅保留人工建议。"
                suggestions.append({
                    "location": label,
                    "problem": kind,
                    "reason": reason,
                    "original_code": original,
                    "corrected_code": corrected,
                    "before_code": original,
                    "after_code": corrected,
                    "explanation": reason,
                    "requires_confirmation": True,
                    "referenced_columns": referenced_columns,
                    "scope": scope,
                })
    return suggestions


def get_client():
    """按需创建千问客户端，避免未配置密钥时网页无法启动。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError(
            "未检测到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env，"
            "并填写阿里云百炼 API Key。"
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("未安装 openai 依赖；请运行 python -m pip install -r requirements.txt。") from exc

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
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

    relevant_knowledge = knowledge_context or retrieve_relevant_knowledge(
        code_result, data_result, project_context
    )
    compact_context = _compact_project_context(project_context)
    project_overview = (project_context or {}).get("project_overview", {})
    prompt = f"""你是 CodeData-Copilot 的 AI 项目 Review 助手。请基于静态证据、项目概览、数据结果和本地知识库生成专业中文审查报告。

## 输入：项目概览
{json.dumps(project_overview, ensure_ascii=False, indent=2, default=str)}

## 输入：代码分析结果
代码静态检查结果（source_snippet 如有提供，才是用户的原始代码）：
{json.dumps(code_result, ensure_ascii=False, indent=2, default=str)}

## 输入：数据分析结果
{json.dumps(data_result, ensure_ascii=False, indent=2, default=str)}

## 输入：Analyzer 已验证的结构化事实（唯一事实基线）
{json.dumps(analysis_facts or {}, ensure_ascii=False, indent=2, default=str)}

## 输入：轻量检索得到的知识库内容
{relevant_knowledge}

## 输入：项目文件与用户代码上下文
{compact_context}

## 输出格式
严格按以下六个 Markdown 二级标题输出，不要省略任何一项：

## 1. 项目总结
概括项目类型、主要技术和分析范围。

## 2. 发现的问题
按文件或数据集列出检测证据，不把静态风险说成已证实故障。

## 3. 问题原因
解释根因以及对应的知识库原则。

## 4. 影响分析
说明对正确性、泛化、复现性或维护性的潜在影响。

## 5. 优化建议
给出按优先级排列、可以由用户确认后执行的建议。

## 6. 修改代码示例
对每项可修复问题使用“修改前代码 / 修改后代码 / 修改说明”。只可逐字引用已提供的 source_snippet；如未提供，明确写“未提供原始代码片段”，不得编造或声称已经修改文件。

优先使用项目中真实出现的文件名和变量名。涉及预处理时说明训练集拟合、验证/测试集仅 transform 的边界。没有足够证据时明确说明不确定性。不要输出自动覆盖文件的指令。

你不得自行猜测或编造：行数、列名、模型变量、训练/测试状态、评价指标、业务规则、删除比例。
这些事实只能引用“Analyzer 已验证的结构化事实”。如果 Analyzer 未验证，必须写：Unable to verify automatically.
不得生成与 detected_split、detected_metrics、model_features、quantified_impacts 或 formal issues 冲突的结论。"""
    prompt += """

公平性 API 约束：不得自行创造方法名或把不同 metric class 的方法混用。
equal_opportunity_difference 只能在已确认具有真实标签数据集和预测标签数据集的 ClassificationMetric 上给出可执行示例；
BinaryLabelDatasetMetric 只用于数据集公平性指标（如 mean_difference/statistical_parity_difference/disparate_impact）。
若对象类型或所需输入无法确认，只给概念建议并明确写 API Unverified，不生成貌似可执行的调用。"""


    client = get_client()

    response = client.chat.completions.create(

        model="qwen-plus",

        messages=[

            {
                "role":"system",
                "content":
                "你是一个以证据为基础的数据科学项目质量审查 Agent。"
            },

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.3

    )


    content = response.choices[0].message.content
    if not content:
        raise ValueError("千问未返回诊断内容，请稍后重试。")
    return content



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
