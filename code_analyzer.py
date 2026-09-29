"""Conservative AST and notebook checks for data science code review."""

from __future__ import annotations

import ast
import re

import nbformat

from data_flow_analyzer import analyze_notebook_data_flow, analyze_python_data_flow
from notebook_syntax import preprocess_notebook_source


PREPROCESSOR_TYPES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
    "SimpleImputer", "KNNImputer", "IterativeImputer", "PCA", "TruncatedSVD",
    "SelectKBest", "SelectPercentile", "VarianceThreshold", "RFE", "RFECV",
    "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "LabelEncoder",
    "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler",
}
SUPERVISED_PREPROCESSORS = {"SelectKBest", "SelectPercentile", "RFE", "RFECV", "TargetEncoder"}
RANDOM_STATE_ESTIMATORS = {
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier",
    "ExtraTreesRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor",
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor", "KMeans", "MiniBatchKMeans",
    "SGDClassifier", "SGDRegressor", "LogisticRegression", "MLPClassifier", "MLPRegressor",
    "IsolationForest", "RandomizedSearchCV", "train_test_split",
}
CLASSIFIER_TYPES = {
    "LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier", "SVC", "KNeighborsClassifier",
    "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "SGDClassifier", "MLPClassifier",
}
CV_NAMES = {
    "cross_val_score", "cross_validate", "GridSearchCV", "RandomizedSearchCV",
    "KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold",
}
CLASSIFICATION_METRICS = {
    "accuracy_score", "precision_score", "recall_score", "f1_score", "roc_auc_score",
    "classification_report", "confusion_matrix", "log_loss", "average_precision_score",
}
REGRESSION_METRICS = {
    "r2_score", "mean_squared_error", "mean_absolute_error", "root_mean_squared_error",
    "mean_absolute_percentage_error", "median_absolute_error",
}
FAIRNESS_DATASET_METHODS = {
    "mean_difference", "statistical_parity_difference", "disparate_impact",
}
FAIRNESS_PREDICTION_METHODS = {
    "equal_opportunity_difference", "average_odds_difference",
    "average_abs_odds_difference", "true_positive_rate_difference",
}
PATH_CALLS = {
    "open", "Path", "read_csv", "read_excel", "read_parquet", "read_json", "read_pickle",
    "to_csv", "to_excel", "to_parquet", "save", "load", "dump", "joblib.load", "joblib.dump",
    "torch.load", "torch.save", "read", "write",
}
SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:api_?key|secret|token|password|passwd|access_?key|private_?key)(?:$|_)", re.I
)
ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?:home|Users|var|tmp|opt)/)")
PLACEHOLDER_SECRET_RE = re.compile(r"(?:your[_ -]?|example|placeholder|xxx|changeme|<.*>)", re.I)
FALLBACK_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b(?P<name>[A-Za-z_]\w*)\s*=\s*)(?P<quote>['\"])(?P<value>[^'\"\r\n]+)(?P=quote)"
)


def read_notebook(notebook_path):
    """Read a notebook and return code-cell sources for existing callers."""
    with open(notebook_path, "r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def _call_name(node):
    target = node.func if isinstance(node, ast.Call) else node
    parts = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def _receiver_name(call):
    if isinstance(call.func, ast.Attribute):
        value = call.func.value
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            return _call_name(value)
    return None


def _names(node):
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)} if node else set()


def _constant_strings(node):
    values = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return values


def _snippet(code, node):
    snippet = ast.get_source_segment(code, node) if node is not None else None
    if snippet:
        return snippet.strip()
    if node is not None and getattr(node, "lineno", None):
        lines = code.splitlines()
        index = node.lineno - 1
        if 0 <= index < len(lines):
            return lines[index].strip()
    return None


def _issue(
    rule_id,
    category,
    issue_type,
    severity,
    confidence,
    message,
    recommendation,
    node,
    code,
    *,
    source_snippet=None,
    evidence=None,
    **extra,
):
    snippet = source_snippet if source_snippet is not None else _snippet(code, node)
    item = {
        "rule_id": rule_id,
        "category": category,
        "type": issue_type,
        "severity": severity,
        "confidence": confidence,
        "message": message,
        "recommendation": recommendation,
        "line": getattr(node, "lineno", None) if node is not None else None,
        "end_line": getattr(node, "end_lineno", None) if node is not None else None,
        "source_snippet": snippet,
        "evidence": evidence or snippet or message,
    }
    item.update(extra)
    return item


def _keyword(call, name):
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _has_keyword(call, name):
    return any(keyword.arg == name for keyword in call.keywords)


def _assignment_name(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            return target.id
    return None


def _subscript_source(node):
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    values = _constant_strings(node.slice)
    return (node.value.id, values) if values else None


def _is_classification(tree, constructors):
    if any(name in CLASSIFIER_TYPES or name.endswith("Classifier") for name in constructors.values()):
        return True
    called = {_call_name(node).split(".")[-1] for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return bool(called & CLASSIFICATION_METRICS)


def _inside_loop(node, parents):
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.For, ast.While, ast.comprehension)):
            return True
        current = parents.get(current)
    return False


def _ancestor(node, parents, kinds):
    current = parents.get(node)
    while current is not None:
        if isinstance(current, kinds):
            return current
        current = parents.get(current)
    return None


def _custom_transformer_refits(tree, parents):
    """Return stateful preprocessor refits occurring inside custom transform methods."""
    findings = []
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        methods = {node.name: node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        base_names = {_call_name(base).split(".")[-1] for base in class_node.bases}
        is_transformer = bool(
            {"BaseEstimator", "TransformerMixin"} & base_names
            or {"fit", "transform"}.issubset(methods)
        )
        if not is_transformer or "transform" not in methods:
            continue
        fit_method = methods.get("fit")
        stores_state = bool(fit_method and any(
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            and any(
                isinstance(candidate, ast.Attribute)
                and isinstance(candidate.value, ast.Name)
                and candidate.value.id == "self"
                for candidate in ast.walk(node)
            )
            for node in ast.walk(fit_method)
        ))
        for call in (node for node in ast.walk(methods["transform"]) if isinstance(node, ast.Call)):
            method = _call_name(call).split(".")[-1]
            if method not in {"fit", "fit_transform"}:
                continue
            receiver_text = _call_name(call.func.value) if isinstance(call.func, ast.Attribute) else ""
            constructor_refit = isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call)
            known_stateful = any(name in receiver_text for name in PREPROCESSOR_TYPES)
            if constructor_refit:
                known_stateful = _call_name(call.func.value).split(".")[-1] in PREPROCESSOR_TYPES
            if known_stateful or constructor_refit:
                findings.append({
                    "class": class_node.name,
                    "call": call,
                    "fit_stores_state": stores_state,
                    "method": method,
                })
    return findings


def _is_with_open(call, tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            if any(candidate is call for candidate in ast.walk(item.context_expr)):
                return True
    return False


def _redact_secret(value):
    if len(value) <= 6:
        return "***"
    return value[:3] + "..." + value[-3:]


def redact_sensitive_code(code):
    """Redact obvious credential literals before code excerpts can enter an AI prompt."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        def replace(match):
            if not SECRET_NAME_RE.search(match.group("name")):
                return match.group(0)
            return match.group("prefix") + repr(_redact_secret(match.group("value")))

        return FALLBACK_SECRET_ASSIGNMENT_RE.sub(replace, code)
    lines = code.splitlines(keepends=True)
    replacements = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = _assignment_name(node)
        value = node.value
        if not name or not SECRET_NAME_RE.search(name) or not isinstance(value, ast.Constant):
            continue
        if isinstance(value.value, str) and value.value:
            replacements.append((value.lineno, value.col_offset, value.end_col_offset, _redact_secret(value.value)))
    for line_number, start, end, redacted in sorted(replacements, reverse=True):
        index = line_number - 1
        line = lines[index]
        lines[index] = line[:start] + repr(redacted) + line[end:]
    return "".join(lines)


def analyze_code_with_ast(code):
    """Analyze one Python source unit using ordered AST evidence and conservative heuristics."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        lines = code.splitlines()
        snippet = lines[exc.lineno - 1].strip() if exc.lineno and exc.lineno <= len(lines) else None
        return [_issue(
            "CODE_SYNTAX_ERROR", "Correctness", "Syntax Error", "High", "High",
            f"代码无法解析：{exc.msg}",
            "修复语法错误后重新运行静态分析。",
            None, code, source_snippet=snippet, evidence=f"line {exc.lineno}, column {exc.offset}: {exc.msg}",
            line=exc.lineno, end_line=getattr(exc, "end_lineno", exc.lineno), column=exc.offset,
        )]

    issues = []
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Call)), key=lambda node: (node.lineno, node.col_offset))
    transformer_refits = _custom_transformer_refits(tree, parents)
    transformer_refit_calls = {item["call"] for item in transformer_refits}
    assignments = {}
    constructors = {}
    dataframe_vars = set()
    target_sources = {}
    feature_sources = {}
    concat_vars = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = _assignment_name(node)
        if not name:
            continue
        assignments[name] = node.value
        if isinstance(node.value, ast.Call):
            constructor = _call_name(node.value).split(".")[-1]
            constructors[name] = constructor
            full_name = _call_name(node.value)
            if constructor == "DataFrame" or constructor.startswith("read_") or full_name.startswith("pd.read_"):
                dataframe_vars.add(name)
            if constructor == "concat" and {value.casefold() for value in _constant_strings(node.value)}:
                concat_vars.add(name)
            if constructor == "concat":
                concat_names = _names(node.value)
                if any("train" in value.casefold() for value in concat_names) and any(
                    "test" in value.casefold() for value in concat_names
                ):
                    concat_vars.add(name)
        source = _subscript_source(node.value)
        if source:
            frame, columns = source
            if name.casefold() in {"y", "target", "label"} and len(columns) == 1:
                target_sources[name] = (frame, columns[0])
            if name.casefold().startswith("x"):
                feature_sources[name] = (frame, set(columns))
        elif isinstance(node.value, ast.Name) and name.casefold().startswith("x"):
            feature_sources[name] = (node.value.id, None)
        elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr == "drop" and isinstance(node.value.func.value, ast.Name):
                dropped = set(_constant_strings(_keyword(node.value, "columns") or node.value))
                feature_sources[name] = (node.value.func.value.id, {"__dropped__", *dropped})

    preprocess_vars = {name for name, kind in constructors.items() if kind in PREPROCESSOR_TYPES}
    model_vars = {
        name for name, kind in constructors.items()
        if kind not in PREPROCESSOR_TYPES and (
            kind in CLASSIFIER_TYPES or kind.endswith(("Classifier", "Regressor"))
            or kind in {"LinearRegression", "Ridge", "Lasso", "ElasticNet", "SVC", "SVR", "KMeans"}
        )
    }
    classification = _is_classification(tree, constructors)
    split_calls = [call for call in calls if _call_name(call).split(".")[-1] == "train_test_split"]
    split_line = min((call.lineno for call in split_calls), default=None)
    fit_calls = []
    predict_calls = []
    metric_calls = []
    fairness_calls = []

    for finding in transformer_refits:
        call = finding["call"]
        state_note = (
            "fit() 保存了部分状态，但 transform() 仍重新拟合预处理器。"
            if finding["fit_stores_state"] else
            "fit() 未保存可复用状态，transform() 又根据当前输入重新学习映射。"
        )
        issues.append(_issue(
            "CODE_TRANSFORMER_REFIT_DURING_TRANSFORM", "Preprocessing",
            "Custom Transformer Re-fits Encoder During Transform", "High", "High",
            f"自定义 Transformer {finding['class']} 在 transform() 中调用 {finding['method']}()。{state_note}",
            "在 fit() 中拟合并保存为 self.encoder_ 等状态；transform() 只能调用已拟合对象的 transform()。",
            call, code,
            root_cause_id=f"custom-transformer-refit:{finding['class']}:{call.lineno}",
            impacts=["inconsistent category mapping", "validation/test instability"],
            issue_nature="State Lifecycle Error", action="Fix",
        ))

    for call in calls:
        full_name = _call_name(call)
        short_name = full_name.split(".")[-1]
        receiver = _receiver_name(call)
        argument_names = set().union(*(_names(arg) for arg in call.args)) if call.args else set()

        if short_name in {"fit", "fit_transform", "fit_resample"}:
            is_preprocessor = receiver in preprocess_vars or short_name in {"fit_transform", "fit_resample"}
            looks_like_model = (
                receiver in model_vars
                or (receiver or "").casefold() in {"model", "clf", "classifier", "regressor", "estimator", "pipeline", "pipe"}
                or constructors.get(receiver) in {"Pipeline", "GridSearchCV", "RandomizedSearchCV"}
            )
            if short_name == "fit" and not is_preprocessor and looks_like_model:
                fit_calls.append(call)
            has_test_argument = any("test" in name.casefold() for name in argument_names)
            if has_test_argument:
                if is_preprocessor:
                    issues.append(_issue(
                        "CODE_PREPROCESSOR_FIT_TEST", "Data Leakage", "Preprocessing Fit on Test Data",
                        "High", "High",
                        "预处理器在测试集变量上执行了拟合。",
                        "仅在训练集上 fit，并对验证/测试集只调用 transform。",
                        call, code,
                    ))
                else:
                    issues.append(_issue(
                        "CODE_MODEL_FIT_TEST", "Data Leakage", "Test Set Used During Training",
                        "High", "High",
                        "模型 fit 调用显式使用了测试集变量。",
                        "训练阶段只使用训练集，测试集仅用于最终一次独立评价。",
                        call, code,
                    ))
            if (is_preprocessor and short_name != "fit_resample" and not has_test_argument
                    and call not in transformer_refit_calls):
                uses_train = any("train" in name.casefold() for name in argument_names)
                uses_full = any(name.casefold() in {"x", "y", "df", "data", "dataset"} for name in argument_names)
                if split_line and call.lineno < split_line and not uses_train:
                    issues.append(_issue(
                        "CODE_PREPROCESS_BEFORE_SPLIT", "Data Leakage", "Data Leakage Risk",
                        "High", "High",
                        f"{short_name} 在 train_test_split 之前运行，可能用全量数据拟合预处理步骤。",
                        "先划分数据，再仅用训练集拟合预处理器。",
                        call, code,
                    ))
                elif split_line and uses_full and not uses_train:
                    issues.append(_issue(
                        "CODE_PREPROCESS_FULL_DATA", "Data Leakage", "Data Leakage Risk",
                        "High", "High",
                        "划分存在，但预处理器仍显式在全量变量上拟合。",
                        "将 fit/fit_transform 的输入改为训练集变量。",
                        call, code,
                    ))
                elif split_line is None and uses_full and not uses_train:
                    issues.append(_issue(
                        "CODE_PREPROCESS_FULL_DATA", "Data Leakage", "Data Leakage Risk",
                        "High", "High",
                        "预处理器在未建立训练/测试边界时使用全量数据拟合。",
                        "先划分数据，再仅用训练集 fit，并对验证/测试集只调用 transform。",
                        call, code,
                    ))
            if short_name == "fit_resample" and split_line and call.lineno < split_line:
                issues.append(_issue(
                    "CODE_RESAMPLE_BEFORE_SPLIT", "Data Leakage", "Potential Resampling Leakage",
                    "High", "High",
                    "重采样在数据划分之前执行，验证/测试样本可能影响合成样本。",
                    "先划分训练与测试数据，再仅对训练集执行重采样。",
                    call, code,
                ))
            if argument_names & concat_vars:
                issues.append(_issue(
                    "CODE_TRAIN_TEST_CONCAT_FIT", "Data Leakage", "Train/Test Concatenation Used for Fit",
                    "High", "Medium",
                    "fit 输入来自明显合并 train/test 的变量。",
                    "保持训练与测试数据边界，并只在训练分区拟合。",
                    call, code,
                ))

        if short_name == "dropna":
            subset = _keyword(call, "subset")
            issues.append(_issue(
                "CODE_DROPNA_SAMPLE_LOSS" if subset is None else "CODE_DROPNA_SUBSET",
                "Missing Data", "Missing Value Handling Risk",
                "Medium" if subset is None else "Low", "High" if subset is None else "Medium",
                "直接 dropna 可能造成样本损失。" if subset is None else "dropna 仅针对指定关键字段，仍需核对样本损失。",
                "先量化删除比例和缺失机制；仅在业务允许时删除记录。",
                call, code,
            ))

        if short_name == "fillna" and split_line and call.lineno < split_line:
            source = (_snippet(code, call) or "").casefold()
            if ".mean(" in source or ".median(" in source:
                issues.append(_issue(
                    "CODE_FILLNA_BEFORE_SPLIT", "Data Leakage", "Potential Preprocessing Leakage",
                    "Medium", "High",
                    "在划分前使用全量数据统计量进行缺失值填补。",
                    "先划分数据，并仅从训练集计算填补统计量。",
                    call, code,
                ))

        if short_name == "predict" or short_name == "score":
            predict_calls.append(call)
        if short_name in CLASSIFICATION_METRICS | REGRESSION_METRICS:
            metric_calls.append(call)
        if short_name in FAIRNESS_DATASET_METHODS | FAIRNESS_PREDICTION_METHODS:
            fairness_calls.append(call)

        if short_name in PATH_CALLS or full_name in PATH_CALLS:
            for path in _constant_strings(call):
                if not re.match(r"^[a-z]+://", path, re.I) and ABSOLUTE_PATH_RE.match(path):
                    issues.append(_issue(
                        "CODE_HARD_CODED_PATH", "Reliability", "Hard-coded Absolute Path",
                        "Low", "High",
                        "文件操作使用了机器相关的绝对路径。",
                        "通过配置、环境变量或项目相对路径注入数据位置。",
                        call, code, evidence=f"检测到绝对路径：{path}",
                    ))
                    break

        if short_name in {"eval", "exec"} and isinstance(call.func, ast.Name):
            issues.append(_issue(
                f"CODE_DANGEROUS_{short_name.upper()}", "Security", f"Dangerous {short_name}()",
                "High", "High", f"检测到动态执行函数 {short_name}()。",
                "避免执行未受信任字符串；改用显式解析或受限映射。",
                call, code,
            ))
        if full_name == "os.system":
            issues.append(_issue(
                "CODE_OS_SYSTEM", "Security", "Shell Execution Risk", "High", "High",
                "os.system 会通过系统 shell 执行命令。",
                "使用参数列表形式的 subprocess，并对输入进行严格限制。",
                call, code,
            ))
        if full_name.startswith("subprocess."):
            shell_value = _keyword(call, "shell")
            if isinstance(shell_value, ast.Constant) and shell_value.value is True:
                issues.append(_issue(
                    "CODE_SUBPROCESS_SHELL", "Security", "Subprocess Shell Risk",
                    "High", "High", "subprocess 调用显式设置了 shell=True。",
                    "改用参数列表和 shell=False，并限制所有外部输入。",
                    call, code,
                ))

        if short_name == "iterrows":
            issues.append(_issue(
                "CODE_PANDAS_ITERROWS", "Performance", "Pandas iterrows Performance Warning",
                "Low", "High", "检测到 DataFrame.iterrows() 逐行迭代。",
                "优先使用向量化操作、assign/apply 或批处理；先以性能测试验证收益。",
                call, code,
            ))
        if short_name == "append" and receiver in dataframe_vars:
            issues.append(_issue(
                "CODE_PANDAS_APPEND", "Pandas", "Deprecated Pandas API",
                "Low", "High", "检测到 DataFrame.append()，该 API 已弃用。",
                "使用 pd.concat([...], ignore_index=...) 替代，并验证索引行为。",
                call, code,
            ))

    for split in split_calls:
        if not _has_keyword(split, "random_state"):
            issues.append(_issue(
                "CODE_SPLIT_RANDOM_STATE", "Reproducibility", "Reproducibility Risk",
                "Low", "High", "train_test_split 未设置 random_state。",
                "为可复现的实验设置显式 random_state，并记录其值。",
                split, code,
            ))
        if classification and not _has_keyword(split, "stratify"):
            issues.append(_issue(
                "CODE_UNSTRATIFIED_SPLIT", "Validation", "Potential Unstratified Split",
                "Low", "Medium", "分类任务的数据划分未显式设置 stratify。",
                "若类别分布重要，使用 stratify=y 并验证各分区分布。",
                split, code,
            ))
        test_size = _keyword(split, "test_size")
        if isinstance(test_size, ast.Constant) and isinstance(test_size.value, (int, float)):
            value = float(test_size.value)
            if 0 < value < 0.05:
                issues.append(_issue(
                    "CODE_INSUFFICIENT_TEST_SAMPLE", "Validation", "Insufficient Final Test Sample",
                    "Low", "Medium",
                    f"test_size={value:g} 可能使最终测试集过小，评价不稳定。",
                    "结合总样本量、类别分布和置信区间确认最终测试集至少具有足够代表性。",
                    split, code,
                ))

    for name, (frame, target_column) in target_sources.items():
        for feature_name, (feature_frame, columns) in feature_sources.items():
            if frame != feature_frame:
                continue
            target_in_features = columns is None or (
                columns is not None and "__dropped__" not in columns and target_column in columns
            )
            if columns and "__dropped__" in columns and target_column not in columns:
                target_in_features = True
            if not target_in_features:
                continue
            for fit in fit_calls:
                fit_names = set().union(*(_names(arg) for arg in fit.args)) if fit.args else set()
                if feature_name in fit_names and name in fit_names:
                    issues.append(_issue(
                        "CODE_TARGET_LEAKAGE", "Data Leakage", "Potential Target Leakage",
                        "High", "High",
                        f"目标列 {target_column!r} 可能仍保留在特征变量 {feature_name} 中。",
                        f"构造 {feature_name} 时显式排除目标列，并验证训练矩阵字段。",
                        fit, code,
                        evidence=f"{name} 来自 {frame}[{target_column!r}]，而 {feature_name} 仍引用同一数据框。",
                    ))

    if fit_calls and not split_calls:
        issues.append(_issue(
            "CODE_NO_VALIDATION_SPLIT", "Validation", "No Explicit Validation Split",
            "Medium", "Medium", "检测到模型训练，但未发现明确的训练/验证或训练/测试划分。",
            "建立独立验证边界，或使用适当的交叉验证策略。",
            fit_calls[0], code,
        ))
    if fit_calls and not predict_calls and not metric_calls and not fairness_calls:
        issues.append(_issue(
            "CODE_NO_EVALUATION", "Evaluation", "No Explicit Evaluation",
            "Low", "Medium", "模型 fit 后未发现 predict、score 或常见评价指标。",
            "在独立验证/测试集上记录与任务目标一致的评价指标。",
            fit_calls[-1], code,
        ))
    if predict_calls:
        predicts_train = any(any("train" in name.casefold() for name in _names(call)) for call in predict_calls)
        predicts_holdout = any(any(
            marker in name.casefold() for name in _names(call) for marker in ("test", "val", "valid")
        ) for call in predict_calls)
        if predicts_train and not predicts_holdout:
            issues.append(_issue(
                "CODE_TRAINING_ONLY_EVALUATION", "Evaluation", "Training-only Evaluation",
                "Medium", "High", "预测/评价仅明显使用训练集变量，未发现独立留出集评价。",
                "在未参与拟合的验证或测试集上报告性能。",
                predict_calls[0], code,
            ))
    called_metrics = {_call_name(call).split(".")[-1] for call in metric_calls}
    if called_metrics == {"accuracy_score"}:
        issues.append(_issue(
            "CODE_SINGLE_CLASSIFICATION_METRIC", "Evaluation", "Single Metric Evaluation",
            "Low", "High", "分类评价仅发现 accuracy_score。",
            "根据业务代价补充 precision/recall/F1、ROC-AUC 或 PR-AUC 等适用指标。",
            metric_calls[0], code,
        ))
    if len(called_metrics) == 1 and called_metrics.issubset(REGRESSION_METRICS):
        issues.append(_issue(
            "CODE_SINGLE_REGRESSION_METRIC", "Evaluation", "Single Metric Evaluation",
            "Low", "High", f"回归评价仅发现 {next(iter(called_metrics))}。",
            "结合误差尺度补充 MAE、RMSE、R²、残差分析或业务指标。",
            metric_calls[0], code,
        ))
    if fit_calls and not any(_call_name(call).split(".")[-1] in CV_NAMES for call in calls):
        issues.append(_issue(
            "CODE_NO_CROSS_VALIDATION", "Validation", "No Cross-validation Detected",
            "Low", "Medium", "检测到模型训练，但未发现常见交叉验证调用。",
            "样本和任务允许时，使用与数据结构匹配的交叉验证评估稳定性。",
            fit_calls[0], code,
        ))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            name = _assignment_name(node)
            value = node.value
            if name and SECRET_NAME_RE.search(name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value and not PLACEHOLDER_SECRET_RE.search(value.value):
                    redacted = f"{name} = {_redact_secret(value.value)!r}"
                    issues.append(_issue(
                        "CODE_HARD_CODED_SECRET", "Security", "Hard-coded Credential",
                        "High", "High", "检测到疑似凭据被直接写入源代码。",
                        "立即轮换暴露的凭据，并改用环境变量或受控 secrets 管理。",
                        node, code, source_snippet=redacted,
                        evidence=f"变量 {name} 被赋予字符串字面量（内容已脱敏）。",
                    ))
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            issues.append(_issue(
                "CODE_WILDCARD_IMPORT", "Maintainability", "Wildcard Import",
                "Low", "High", "通配符导入会隐藏命名来源并增加冲突风险。",
                "显式导入实际使用的名称。", node, code,
            ))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [value for value in node.args.kw_defaults if value]
            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(_issue(
                        "CODE_MUTABLE_DEFAULT", "Reliability", "Mutable Default Argument",
                        "Medium", "High", f"函数 {node.name} 使用了可变默认参数。",
                        "默认值改为 None，并在函数内部创建新容器。",
                        default, code,
                    ))
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > 100:
                issues.append(_issue(
                    "CODE_LONG_FUNCTION", "Maintainability", "Very Long Function",
                    "Low", "High", f"函数 {node.name} 长度为 {length} 行。",
                    "按单一职责拆分，并保留清晰的数据边界与测试。",
                    node, code, evidence=f"函数跨度 {length} 行。",
                ))
            complexity = sum(isinstance(item, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.Match)) for item in ast.walk(node))
            if complexity > 15:
                issues.append(_issue(
                    "CODE_BRANCH_COMPLEXITY", "Maintainability", "Excessive Branch Complexity",
                    "Low", "High", f"函数 {node.name} 包含较多控制分支。",
                    "提取独立规则或步骤，降低单个函数的认知复杂度。",
                    node, code, evidence=f"检测到 {complexity} 个分支/循环结构。",
                ))
        if isinstance(node, ast.ExceptHandler):
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception")
            swallowed = broad and len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
            if swallowed:
                issues.append(_issue(
                    "CODE_SWALLOWED_EXCEPTION", "Reliability", "Swallowed Exception",
                    "Medium", "High", "宽泛异常被 pass 静默吞掉。",
                    "捕获具体异常并记录上下文，或在无法恢复时重新抛出。",
                    node, code,
                ))
            elif broad:
                issues.append(_issue(
                    "CODE_BROAD_EXCEPTION", "Reliability", "Broad Exception Handling",
                    "Low", "High", "检测到 bare except 或宽泛 Exception 捕获。",
                    "优先捕获具体异常类型，并保留日志或恢复策略。",
                    node, code,
                ))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript):
                    issues.append(_issue(
                        "CODE_CHAINED_ASSIGNMENT", "Pandas", "Potential Chained Assignment",
                        "Medium", "High", "检测到对连续索引表达式进行赋值。",
                        "使用 df.loc[rows, column] = value，并确认是否需要 copy()。",
                        node, code,
                    ))

    for call in calls:
        if _call_name(call).split(".")[-1] == "open" and not _is_with_open(call, tree):
            issues.append(_issue(
                "CODE_OPEN_NO_CONTEXT", "Reliability", "Open Without Context Manager",
                "Low", "High", "open() 未明显位于 with 上下文管理器中。",
                "使用 with open(...) as handle，确保文件句柄可靠关闭。",
                call, code,
            ))

    for name, constructor in constructors.items():
        if constructor not in RANDOM_STATE_ESTIMATORS:
            continue
        call = assignments[name]
        if isinstance(call, ast.Call) and not _has_keyword(call, "random_state"):
            issues.append(_issue(
                "CODE_MODEL_RANDOM_STATE", "Reproducibility", "Model Random State Missing",
                "Low", "High", f"{constructor} 未设置 random_state。",
                "设置并记录 random_state，以便复现实验结果。",
                call, code,
            ))

    random_calls = [
        call for call in calls
        if _call_name(call).startswith(("np.random.", "numpy.random.", "random.", "torch."))
        and _call_name(call).split(".")[-1] not in {"seed", "manual_seed", "set_seed"}
    ]
    seeded = any(_call_name(call).endswith((".seed", ".manual_seed", ".set_seed")) for call in calls)
    if random_calls and not seeded:
        issues.append(_issue(
            "CODE_RANDOM_SEED", "Reproducibility", "Random Seed Missing",
            "Low", "Medium", "代码使用了随机操作，但未发现显式 seed 设置。",
            "集中设置并记录 Python、NumPy 及所用框架的随机种子。",
            random_calls[0], code,
        ))

    for call in metric_calls:
        if _inside_loop(call, parents) and any("test" in name.casefold() for name in _names(call)):
            issues.append(_issue(
                "CODE_TEST_SET_OVERUSE", "Evaluation", "Test-set Overuse / Model Selection Leakage",
                "High", "High", "循环或搜索过程中反复使用测试集变量比较候选模型或超参数。",
                "使用验证集进行选择和调参，仅在最终方案锁定后评估测试集。",
                call, code,
                root_cause_id=f"test-set-model-selection:{call.lineno}",
            ))
        if _inside_loop(call, parents) and any(
            marker in name.casefold() for name in _names(call) for marker in ("val", "valid")
        ):
            issues.append(_issue(
                "CODE_VALIDATION_OVERFITTING_RISK", "Evaluation", "Validation Overfitting Risk",
                "Medium", "Medium", "循环或搜索过程反复在同一验证集上比较候选配置。",
                "限制搜索空间，采用嵌套交叉验证或保留独立最终测试集，并记录选择次数。",
                call, code,
            ))

    # Collapse only exact duplicate findings; retain distinct lines and evidence.
    deduplicated = []
    seen = set()
    for issue in sorted(issues, key=lambda item: (item.get("line") or 0, item["rule_id"])):
        key = (issue["rule_id"], issue.get("line"), issue.get("source_snippet"))
        if key not in seen:
            deduplicated.append(issue)
            seen.add(key)
    return deduplicated


def _notebook_issue(
    rule_id, issue_type, severity, confidence, message, recommendation,
    cell=None, evidence=None, line=None, **extra,
):
    return {
        "rule_id": rule_id,
        "category": "Notebook",
        "type": issue_type,
        "severity": severity,
        "confidence": confidence,
        "message": message,
        "recommendation": recommendation,
        "line": line,
        "end_line": line,
        "source_snippet": None,
        "evidence": evidence or message,
        "notebook_cell": cell,
        **extra,
    }


def _strip_notebook_magics(source):
    return preprocess_notebook_source(source)["code"]


def analyze_notebook(notebook_path):
    """Analyze code cells plus execution-order and output metadata."""
    with open(notebook_path, "r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    results = []
    execution = []
    code_cells = [(index, cell) for index, cell in enumerate(notebook.cells, 1) if cell.cell_type == "code"]

    for cell_index, cell in code_cells:
        if isinstance(cell.execution_count, int):
            execution.append((cell_index, cell.execution_count))
        notebook_syntax = preprocess_notebook_source(cell.source)
        cell_issues = analyze_code_with_ast(notebook_syntax["code"])
        for construct in notebook_syntax["constructs"]:
            source = construct.get("source", "")
            if re.match(r"^\s*!\s*(?:pip|conda)\s+install\b", source, re.I):
                continue
            cell_issues.append(_notebook_issue(
                "NB_IPYTHON_MAGIC" if construct["kind"] == "IPython Magic" else
                "NB_SHELL_COMMAND" if construct["kind"] == "Shell Command" else
                "NB_NOTEBOOK_HELP_SYNTAX",
                construct["kind"], "Low", "High",
                f"检测到 {construct['kind']}；已在 Python AST 分析前按 Notebook 语法处理。",
                "该语法不作为 Python Syntax Error；导出为 .py 前需转换为普通 Python。",
                cell=cell_index, line=construct.get("line"), evidence=source,
                issue_nature="Notebook-specific Syntax", action="Informational", score_impact=0,
            ))
        line_count = len(cell.source.splitlines())
        if line_count > 150:
            cell_issues.append(_notebook_issue(
                "NB_LARGE_CELL", "Large Notebook Cell", "Low", "High",
                f"Notebook Cell {cell_index} 包含 {line_count} 行代码。",
                "将数据加载、特征处理、训练和评价拆分为可复查步骤。",
                cell=cell_index, evidence=f"cell={cell_index}, lines={line_count}",
            ))
        if re.search(r"^\s*!\s*(?:pip|conda)\s+install\b", cell.source, re.M | re.I):
            cell_issues.append(_notebook_issue(
                "NB_INLINE_INSTALL", "Environment Reproducibility Issue", "Low", "High",
                "Notebook 中直接执行了 pip/conda install。",
                "将依赖固定到 requirements/环境文件，并在 Notebook 中说明环境版本。",
                cell=cell_index, evidence="检测到内联依赖安装命令。",
                issue_nature="Shell Command", action="Review",
            ))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                cell_issues.append(_notebook_issue(
                    "NB_EXECUTION_ERROR", "Notebook Execution Error", "High", "High",
                    f"Notebook Cell {cell_index} 保存了错误输出。",
                    "修复错误并从干净内核按顺序重新运行全部单元格。",
                    cell=cell_index,
                    evidence=f"{output.get('ename', 'Error')}: {output.get('evalue', '')}",
                ))
        if cell_issues:
            results.append({"cell": cell_index, "issues": cell_issues})

    counts = [count for _, count in execution]
    if any(current <= previous for previous, current in zip(counts, counts[1:])):
        offending = next(
            cell_index for (cell_index, current), (_, previous) in zip(execution[1:], execution)
            if current <= previous
        )
        results.append({"cell": offending, "issues": [_notebook_issue(
            "NB_NON_LINEAR_EXECUTION", "Non-linear Notebook Execution", "Medium", "High",
            "Notebook 执行计数不是严格递增，结果可能依赖隐藏状态。",
            "重启内核并从上到下运行所有单元格，确保结果可复现。",
            cell=offending, evidence=f"execution_count sequence={counts}",
        )]})
    executed_seen = False
    for cell_index, cell in code_cells:
        if isinstance(cell.execution_count, int):
            executed_seen = True
        elif executed_seen and cell.source.strip():
            results.append({"cell": cell_index, "issues": [_notebook_issue(
                "NB_UNEXECUTED_AFTER_EXECUTED", "Potential Hidden State Risk", "Low", "Low",
                "已执行单元格之后存在未执行代码单元格。",
                "从干净内核顺序运行并确认未执行单元格是否属于当前工作流。",
                cell=cell_index, evidence=f"Cell {cell_index} execution_count=None",
            )]})
            break
    # Cell-local rules cannot see a later split or metric. Reconcile only absence claims
    # against the notebook-level flow while preserving concrete line-level findings.
    flow = analyze_notebook_data_flow(notebook_path)
    split_detected = bool(flow.get("split", {}).get("detected"))
    evaluation_detected = bool(flow.get("evaluation_detected"))
    reconciled = []
    for result in results:
        issues = [
            issue for issue in result.get("issues", [])
            if not (
                (split_detected and issue.get("rule_id") == "CODE_NO_VALIDATION_SPLIT")
                or (evaluation_detected and issue.get("rule_id") == "CODE_NO_EVALUATION")
            )
        ]
        if issues:
            reconciled.append({**result, "issues": issues})
    return reconciled


def analyze_code_unit(code, file_name=None):
    """Return the stable issue list and its project-flow facts together."""
    return {
        "issues": analyze_code_with_ast(code),
        "data_flow": analyze_python_data_flow(code, file_name=file_name),
    }


def print_result(notebook_path):
    results = analyze_notebook(notebook_path)
    print("=" * 40)
    if not results:
        print("未发现明显问题")
    for result in results:
        print(f"\nCell {result.get('cell', 'Notebook')}")
        for issue in result["issues"]:
            print(f"{issue['severity']} · {issue['type']} · {issue['message']}")
    print("=" * 40)


if __name__ == "__main__":
    print_result("demo_data/test.ipynb")
