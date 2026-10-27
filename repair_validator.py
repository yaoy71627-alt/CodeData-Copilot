"""Conservative syntax, context, and known-API validation for repair snippets."""

from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap

import nbformat


API_REGISTRY = {
    "BinaryLabelDatasetMetric": {
        "methods": {
            "mean_difference", "statistical_parity_difference", "disparate_impact",
            "consistency", "smoothed_empirical_differential_fairness", "num_instances",
        },
        "minimum_constructor_inputs": 1,
    },
    "ClassificationMetric": {
        "methods": {
            "equal_opportunity_difference", "average_odds_difference",
            "average_abs_odds_difference", "true_positive_rate_difference",
            "false_positive_rate_difference", "error_rate_difference",
            "mean_difference", "statistical_parity_difference", "disparate_impact",
        },
        "minimum_constructor_inputs": 2,
    },
    "Reweighing": {
        "methods": {"fit", "transform", "fit_transform"},
        "minimum_constructor_inputs": 0,
    },
    "DataFrame": {
        "methods": {"fillna", "dropna", "drop_duplicates", "copy", "astype", "replace"},
        "minimum_constructor_inputs": 0,
    },
    "Series": {
        "methods": {"fillna", "dropna", "astype", "replace", "median", "mean", "mode"},
        "minimum_constructor_inputs": 0,
    },
    "LabelEncoder": {
        "methods": {"fit", "transform", "fit_transform", "inverse_transform"},
        "minimum_constructor_inputs": 0,
    },
    "OneHotEncoder": {
        "methods": {"fit", "transform", "fit_transform", "inverse_transform"},
        "minimum_constructor_inputs": 0,
    },
    "StandardScaler": {
        "methods": {"fit", "transform", "fit_transform", "inverse_transform"},
        "minimum_constructor_inputs": 0,
    },
    "MinMaxScaler": {
        "methods": {"fit", "transform", "fit_transform", "inverse_transform"},
        "minimum_constructor_inputs": 0,
    },
    "SimpleImputer": {
        "methods": {"fit", "transform", "fit_transform", "inverse_transform"},
        "minimum_constructor_inputs": 0,
    },
}
RUNTIME_CLASS_PATHS = {
    "BinaryLabelDatasetMetric": ("aif360.metrics", "BinaryLabelDatasetMetric"),
    "ClassificationMetric": ("aif360.metrics", "ClassificationMetric"),
    "Reweighing": ("aif360.algorithms.preprocessing", "Reweighing"),
    "DataFrame": ("pandas", "DataFrame"),
    "Series": ("pandas", "Series"),
    "LabelEncoder": ("sklearn.preprocessing", "LabelEncoder"),
    "OneHotEncoder": ("sklearn.preprocessing", "OneHotEncoder"),
    "StandardScaler": ("sklearn.preprocessing", "StandardScaler"),
    "MinMaxScaler": ("sklearn.preprocessing", "MinMaxScaler"),
    "SimpleImputer": ("sklearn.impute", "SimpleImputer"),
}
ESTIMATOR_CLASS_PATHS = {
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "MLPClassifier": ("sklearn.neural_network", "MLPClassifier"),
    "MLPRegressor": ("sklearn.neural_network", "MLPRegressor"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "SVC": ("sklearn.svm", "SVC"),
    "SVR": ("sklearn.svm", "SVR"),
}


def _flow_context_for_suggestion(data_flow, suggestion):
    candidates = list((data_flow or {}).get("files") or [data_flow or {}])
    file_name = suggestion.get("file")
    if file_name:
        matching = [flow for flow in candidates if flow.get("file") == file_name]
        if matching:
            return matching[0]
    return data_flow or {}


def _defined_variables(flow, suggestion):
    insertion_line = suggestion.get("line")
    insertion_cell = suggestion.get("cell")
    variables = set()
    for name, locations in (flow.get("definitions") or {}).items():
        for location in locations:
            if insertion_cell is None:
                visible = (
                    location.get("cell") is None
                    and (insertion_line is None or (location.get("line") or 0) < insertion_line)
                )
            else:
                location_cell = location.get("cell")
                visible = (
                    location_cell is not None
                    and (
                        location_cell < insertion_cell
                        or (
                            location_cell == insertion_cell
                            and (insertion_line is None or (location.get("line") or 0) < insertion_line)
                        )
                    )
                )
            if visible:
                variables.add(name)
                break
    if not flow.get("definitions"):
        # Backward compatibility for third-party/serialized flows created
        # before definition locations were recorded.
        variables.update((flow.get("object_types") or {}).keys())
        variables.update((flow.get("variable_columns") or {}).keys())
        variables.update(
            source.get("variable") for source in flow.get("source_data", []) if source.get("variable")
        )
        split = flow.get("split", {})
        variables.update(split.get("train", []))
        variables.update(split.get("test", []))
        for names in split.get("split_roles", {}).values():
            variables.update(names)
    variables.discard(None)
    return variables


def _snippet_references(tree):
    """Return external data variables and dataframe-column references."""
    first_assignment = {}
    parameters = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            parameters.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            first_assignment[node.id] = min(
                first_assignment.get(node.id, float("inf")), getattr(node, "lineno", 0)
            )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bindings = [alias.asname or alias.name.split(".")[0] for alias in node.names]
            imported.update(bindings)
            for name in bindings:
                first_assignment[name] = min(first_assignment.get(name, float("inf")), node.lineno)
        elif isinstance(node, ast.ImportFrom):
            bindings = [alias.asname or alias.name for alias in node.names]
            imported.update(bindings)
            for name in bindings:
                first_assignment[name] = min(first_assignment.get(name, float("inf")), node.lineno)
    loaded_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        assigned_line = first_assignment.get(node.id)
        if node.id not in parameters and (
            assigned_line is None or getattr(node, "lineno", 0) <= assigned_line
        ):
            loaded_names.add(node.id)
    columns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                columns.setdefault(node.value.id, set()).add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name):
                if node.func.attr == "fillna" and node.args and isinstance(node.args[0], ast.Dict):
                    for key in node.args[0].keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            columns.setdefault(receiver.id, set()).add(key.value)
    return loaded_names, columns


def _defined_names(tree):
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _loaded_names(tree):
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _contextual_patch_validation(flow, suggestion, before, after):
    """Apply a replacement to an in-memory project/notebook copy and reparse it."""
    units = [dict(unit) for unit in (flow.get("code_units") or [])]
    if not units or not before:
        return False, [], "完整源文件/Notebook 单元上下文不可用。"
    target_cell = suggestion.get("cell")
    candidates = [
        index for index, unit in enumerate(units)
        if target_cell is None or unit.get("cell") == target_cell
    ]
    replacement, detail = _source_replacement(flow, suggestion, before, after)
    if replacement is None:
        return False, [], detail
    target_index = next(index for index in candidates if units[index].get("cell") == target_cell)
    units[target_index]["code"] = replacement["after_source"]
    errors = []
    parsed_units = []
    with TemporaryDirectory(prefix="codedata-repair-") as directory:
        temporary_root = Path(directory)
        if any(unit.get("cell") is not None for unit in units):
            notebook = nbformat.v4.new_notebook()
            notebook.cells = [
                nbformat.v4.new_code_cell(str(unit.get("code", "")))
                for unit in units
            ]
            temporary_path = temporary_root / "patched_notebook.ipynb"
            nbformat.write(notebook, temporary_path)
            reloaded_sources = [
                cell.source
                for cell in nbformat.read(temporary_path, as_version=4).cells
                if cell.cell_type == "code"
            ]
        else:
            temporary_path = temporary_root / "patched_source.py"
            temporary_path.write_text(
                "\n\n".join(str(unit.get("code", "")) for unit in units),
                encoding="utf-8",
            )
            reloaded_sources = [str(unit.get("code", "")) for unit in units]
        for index, source in enumerate(reloaded_sources):
            try:
                parsed_units.append(ast.parse(source))
            except SyntaxError as exc:
                unit = units[index]
                errors.append(
                    f"patched unit {unit.get('cell') or index + 1} syntax error: "
                    f"line {exc.lineno}: {exc.msg}"
                )
    if errors:
        return True, errors, "Patch 已应用到临时上下文，但重新解析失败。"

    try:
        before_tree = ast.parse(textwrap.dedent(str(before)))
        after_tree = ast.parse(textwrap.dedent(str(after)))
    except SyntaxError:
        return False, [], "原始片段语法不完整，无法验证定义与下游引用是否兼容。"
    removed_definitions = _defined_names(before_tree) - _defined_names(after_tree)
    if removed_definitions:
        downstream_loads = _loaded_names(parsed_units[target_index])
        for tree in parsed_units[target_index + 1:]:
            downstream_loads.update(_loaded_names(tree))
        broken = sorted(removed_definitions & downstream_loads)
        if broken:
            errors.append(
                "Patch removes definitions still used by downstream code: " + ", ".join(broken)
            )
    return True, errors, (
        "Patch 已应用到临时源文件/Notebook 副本；所有代码单元均通过 AST 重新解析，"
        "且未发现被删除后仍由下游引用的变量。"
        if not errors else
        "Patch 已应用到临时上下文，但下游符号兼容检查失败。"
    )


def _runtime_has_method(object_type, method):
    """Use trusted installed packages for introspection; never import user modules."""
    path = RUNTIME_CLASS_PATHS.get(object_type) or ESTIMATOR_CLASS_PATHS.get(object_type)
    if not path:
        return None
    module_name, class_name = path
    root_name = module_name.split(".")[0]
    if importlib.util.find_spec(root_name) is None:
        return None
    try:
        class_object = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError, RuntimeError):
        return None
    return hasattr(class_object, method)


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
    if not isinstance(call.func, ast.Attribute):
        return None
    value = call.func.value
    if isinstance(value, ast.Name):
        return value.id
    return None


def _direct_constructor_type(call):
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Call):
        return None, None
    constructor = _call_name(call.func.value).split(".")[-1]
    return constructor, call.func.value


def _api_validation(tree, data_flow, suggestion):
    object_types = {
        **((data_flow or {}).get("object_types", {})),
    }
    constructor_inputs = {
        **((data_flow or {}).get("constructor_inputs", {})),
    }
    checked = []
    unverified = []
    failures = []

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute):
            continue
        method = call.func.attr
        receiver = _receiver_name(call)
        object_type = object_types.get(receiver)
        if receiver in API_REGISTRY:
            object_type = receiver
        direct_type, direct_constructor = _direct_constructor_type(call)
        if direct_type:
            object_type = direct_type
        if object_type not in API_REGISTRY:
            # Estimators are known trusted classes, not user-provided imports.
            # A nonexistent method is a failure; presence alone does not prove
            # signature/argument compatibility and must remain unverified.
            if object_type in ESTIMATOR_CLASS_PATHS and _runtime_has_method(object_type, method) is False:
                failures.append(f"{object_type}.{method}() 未通过已安装库的 hasattr introspection")
            else:
                unverified.append(f"{receiver or 'expression'}.{method}() 的对象类型或 API 无法确认")
            continue
        allowed = API_REGISTRY[object_type]["methods"]
        label = f"{object_type}.{method}()"
        if method not in allowed:
            failures.append(f"{label} 不在已知 API registry 中")
            continue
        runtime_check = _runtime_has_method(object_type, method)
        if runtime_check is False:
            failures.append(f"{label} 未通过已安装库的 hasattr introspection")
            continue

        minimum = API_REGISTRY[object_type].get("minimum_constructor_inputs", 0)
        provided = None
        if direct_constructor is not None:
            provided = len(direct_constructor.args)
        elif receiver in constructor_inputs:
            inputs = constructor_inputs[receiver]
            provided = len(inputs.get("positional", []))
            if object_type == "ClassificationMetric" and provided < 2:
                keyword_names = set(inputs.get("keywords", {}))
                provided += int("dataset_true" in keyword_names) + int("dataset_pred" in keyword_names)
        if minimum and (provided is None or provided < minimum):
            if object_type == "ClassificationMetric":
                unverified.append(
                    f"{label} 需要可确认的真实标签数据集和预测标签数据集；请先构造 ClassificationMetric"
                )
            else:
                unverified.append(f"{label} 的构造输入无法确认")
            continue
        checked.append(label)
    return checked, unverified, failures


def _literal_dict_keys(node, assignments):
    if isinstance(node, ast.Name):
        node = assignments.get(node.id)
    if not isinstance(node, ast.Dict):
        return None
    keys = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        keys.append(key.value.split("__")[-1])
    return keys


def _estimator_parameters(type_name):
    path = ESTIMATOR_CLASS_PATHS.get(type_name)
    if not path:
        return None
    try:
        estimator_class = getattr(importlib.import_module(path[0]), path[1])
        return set(estimator_class().get_params(deep=True))
    except (ImportError, AttributeError, TypeError, RuntimeError):
        return None


def _grid_search_validation(tree, flow):
    """Reject a shared parameter grid that is invalid for any possible estimator."""
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
    checked, unverified, failures = [], [], []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _call_name(call).split(".")[-1] != "GridSearchCV":
            continue
        estimator_node = call.args[0] if call.args else next(
            (item.value for item in call.keywords if item.arg == "estimator"), None
        )
        grid_node = call.args[1] if len(call.args) > 1 else next(
            (item.value for item in call.keywords if item.arg == "param_grid"), None
        )
        grid_keys = _literal_dict_keys(grid_node, assignments)
        if grid_keys is None:
            unverified.append("GridSearchCV 的 param_grid 不是可静态解析的字面量字典")
            continue
        estimator_types = []
        if isinstance(estimator_node, ast.Call):
            estimator_types = [_call_name(estimator_node).split(".")[-1]]
        elif isinstance(estimator_node, ast.Name):
            name = estimator_node.id
            estimator_types = list((flow.get("possible_object_types") or {}).get(name, []))
            if not estimator_types and (flow.get("object_types") or {}).get(name):
                estimator_types = [(flow.get("object_types") or {})[name]]
        if not estimator_types:
            unverified.append("GridSearchCV 的 estimator 类型无法确认")
            continue
        for estimator_type in estimator_types:
            parameters = _estimator_parameters(estimator_type)
            if parameters is None:
                unverified.append(f"无法读取 {estimator_type} 的参数集合")
                continue
            invalid = sorted(set(grid_keys) - parameters)
            if invalid:
                failures.append(
                    f"GridSearchCV 参数 {invalid} 不适用于候选 estimator {estimator_type}"
                )
            else:
                checked.append(f"GridSearchCV({estimator_type}) parameters")
    return checked, unverified, failures



VALIDATION_STATUSES = frozenset({
    "Recommendation Only", "Syntax Valid Only", "API Unverified",
    "Statically Validated", "Runtime Verified", "Failed Validation",
})


def _source_replacement(flow, suggestion, before, after):
    """Locate exactly at a declared line/cell, never replace the first matching text."""
    units = flow.get("code_units") or []
    line, cell = suggestion.get("line"), suggestion.get("cell")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        return None, "缺少明确的原始行号。"
    if not suggestion.get("file") or flow.get("file") != suggestion["file"]:
        return None, "建议文件与源上下文不匹配，无法确认原始文件。"
    matches = [u for u in units if u.get("cell") == cell]
    if len(matches) != 1 or not before:
        return None, "原始文件/单元上下文不可用或不能唯一定位。"
    source = str(matches[0].get("code", ""))
    lines = source.splitlines(keepends=True)
    if line > len(lines):
        return None, "建议行号超出原始代码范围。"
    start = sum(map(len, lines[:line - 1]))
    indent = len(lines[line - 1]) - len(lines[line - 1].lstrip(" \t"))
    options = [start, start + indent]
    position = next((offset for offset in options if source.startswith(str(before), offset)), None)
    if position is None:
        return None, "原始代码与声明行号处的内容不一致，禁止按其他位置猜测替换。"
    end = position + len(str(before))
    if source[start:position].strip() or source[end:].split("\n", 1)[0].strip():
        return None, "修复未覆盖完整语句边界，不能安全替换局部字符。"
    replacement = str(after)
    if position != start and "\n" in replacement:
        return None, "缩进块中的多行替换需要额外作用域验证。"
    patched = source[:position] + replacement + source[end:]
    return {"file": suggestion["file"], "cell": cell, "line": line,
            "start": position, "end": end, "before_source": source,
            "after_source": patched, "before_code": str(before), "after_code": str(after),
            "scope": "python_file" if cell is None else "notebook_cell_fragment",
            "project_applicable": cell is None}, "已在内存源快照的声明位置应用替换。"


def _schema_columns(flow, data_results):
    """Use only the dataset resolved by this file, never union unrelated schemas."""
    if flow.get('strict_dataset_binding'):
        if flow.get('association_status') != 'matched':
            return set()
        matching = [entry for entry in data_results or [] if entry.get('file') == flow.get('bound_dataset_name')]
        return set(matching[0].get('result', matching[0]).get('data_types', {})) if len(matching) == 1 else set()
    sources = [item.get("path") or item.get("file") for item in flow.get("source_data", [])]
    # The analyzer records the literal path under source as well in older flows.
    sources += [item.get("source") for item in flow.get("source_data", [])]
    sources = {str(s).replace("\\", "/") for s in sources if isinstance(s, str)}
    entries = list(data_results or [])
    if sources:
        matches = [entry for entry in entries if str(entry.get("file", "")).replace("\\", "/") in sources]
        if len(matches) == 1:
            return set(matches[0].get("result", matches[0]).get("data_types", {}))
        return set()
    if len(entries) == 1:
        return set(entries[0].get("result", entries[0]).get("data_types", {}))
    return set()


def _column_references(tree, flow):
    """Resolve literal column selectors and common column-bearing keywords."""
    columns, unresolved = {}, []
    assignments = {n.targets[0].id: n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    def labels(node):
        if isinstance(node, ast.Name):
            node = assignments.get(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [labels(item) for item in node.elts]
            return set().union(*values) if all(value is not None for value in values) else None
        if isinstance(node, ast.Dict):
            values = [labels(key) for key in node.keys]
            return set().union(*values) if all(value is not None for value in values) else None
        return None
    for node in ast.walk(tree):
        receiver, selector = None, None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            receiver, selector = node.value.id, node.slice
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "loc" and isinstance(node.value.value, ast.Name):
            receiver = node.value.value.id
            if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                selector = node.slice.elts[1]
        if receiver and selector is not None:
            values = labels(selector)
            if values is not None:
                columns.setdefault(receiver, set()).update(values)
            elif isinstance(selector, (ast.List, ast.Name, ast.Tuple)):
                unresolved.append(f"{receiver} 的动态字段选择无法确认")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            receiver = node.func.value.id
            selections = [kw.value for kw in node.keywords if kw.arg in {"columns", "subset"}]
            if node.func.attr in {"fillna", "astype"} and node.args and isinstance(node.args[0], ast.Dict):
                selections.append(node.args[0])
            for selector in selections:
                values = labels(selector)
                if values is None:
                    unresolved.append(f"{receiver} 的字段参数无法静态解析")
                else:
                    columns.setdefault(receiver, set()).update(values)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            receiver = node.value.id
            if flow.get("object_types", {}).get(receiver) == "DataFrame":
                known_attributes = API_REGISTRY["DataFrame"]["methods"] | {"loc", "iloc", "shape", "columns", "values", "dtypes", "index", "empty"}
                if node.attr not in known_attributes and not any(isinstance(parent, ast.Call) and parent.func is node for parent in ast.walk(tree)):
                    columns.setdefault(receiver, set()).add(node.attr)
    return columns, unresolved


def _requires_coordinated_repair(item, before, after):
    if item.get("requires_multi_location") or item.get("problem") in {
        "Data Leakage Risk", "Potential Data Leakage", "Missing Value Handling Risk",
        "Test-set Used for Model Comparison", "Test-set Used for Hyperparameter Comparison",
    }:
        return True
    risky = {"fit", "fit_transform", "transform", "train_test_split", "dropna", "fillna", "drop_duplicates"}
    for code in (before, after):
        try:
            tree = ast.parse(textwrap.dedent(str(code or "")))
        except SyntaxError:
            continue
        if any(_call_name(node).split(".")[-1] in risky for node in ast.walk(tree) if isinstance(node, ast.Call)):
            return True
    return False


def _finish_repair(item, status, detail):
    item["validation_status"], item["validation_detail"] = status, detail
    item["runtime_verified"] = False  # No user code is executed by this validator.
    item["executable"] = status == "Statically Validated"
    if status != "Statically Validated":
        item["candidate_code"] = item.get("after_code") or item.get("corrected_code") or item.get("candidate_code")
        item["after_code"] = item["corrected_code"] = None
        item["patch_validation"] = None
    return item



def validate_repair_suggestions(suggestions, data_flow=None, data_results=None):
    """Validate syntax, scope, columns, APIs and exact application without execution."""
    validated = []
    for suggestion in suggestions or []:
        item = deepcopy(suggestion)
        item.pop("patch_validation", None)
        flow = _flow_context_for_suggestion(data_flow, item)
        after = item.get("after_code") or item.get("corrected_code")
        if not after:
            # Revalidation must preserve a previous failure/limited verification,
            # not silently rename it Recommendation Only after code was hidden.
            previous = item.get("validation_status")
            if previous in VALIDATION_STATUSES and previous not in {"Runtime Verified", "Statically Validated"}:
                validated.append(_finish_repair(item, previous, item.get("validation_detail", "未执行运行验证。")))
            else:
                validated.append(_finish_repair(item, "Recommendation Only", "仅提供人工处理方向，没有可执行代码或可导出的 Patch。"))
            continue
        before = item.get("before_code") or item.get("original_code")
        try:
            tree = ast.parse(textwrap.dedent(str(after)))
        except SyntaxError as exc:
            validated.append(_finish_repair(item, "Failed Validation", f"语法校验失败：line {exc.lineno}: {exc.msg}"))
            continue
        failures, unverified, context_unknown = [], [], []
        checked, api_unknown, api_errors = _api_validation(tree, flow, item)
        grid_checked, grid_unknown, grid_errors = _grid_search_validation(tree, flow)
        checked += grid_checked
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
            unverified.append("候选代码中的导入依赖及版本兼容性尚未确认")
        unverified += api_unknown + grid_unknown
        failures += api_errors + grid_errors
        external, _ = _snippet_references(tree)
        allowed = _defined_variables(flow, item) | set(dir(builtins))
        # Import bindings are defined by syntax even when the flow's older
        # definitions schema omitted them. Only top-level imports before target.
        for unit in flow.get("code_units", []):
            if item.get("cell") is not None and (unit.get("cell") or 0) > item["cell"]:
                continue
            try:
                top = ast.parse(unit.get("code", ""))
            except SyntaxError:
                continue
            for node in top.body:
                if unit.get("cell") == item.get("cell") and item.get("line") and node.lineno >= item["line"]:
                    continue
                if isinstance(node, ast.Import):
                    allowed.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    allowed.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        if item.get("line") and flow.get("code_units"):
            top_level = set(dir(builtins))
            for unit in flow["code_units"]:
                if item.get("cell") is not None and (unit.get("cell") or 0) > item["cell"]:
                    continue
                try:
                    body = ast.parse(unit.get("code", "")).body
                except SyntaxError:
                    continue
                for node in body:
                    if unit.get("cell") == item.get("cell") and node.lineno >= item["line"]:
                        continue
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        top_level.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        top_level.add(node.name)
                    elif isinstance(node, ast.Import):
                        top_level.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        top_level.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
            allowed &= top_level
        unknown = external - allowed
        if unknown:
            failures.append("建议位置尚未定义的变量或函数：" + ", ".join(sorted(unknown)))
        source_files = {f.get("file") for f in (data_flow or {}).get("files", []) if f.get("file")}
        if (data_flow or {}).get("file"):
            source_files.add(data_flow["file"])
        if item.get("file") and source_files and item["file"] not in source_files:
            failures.append("建议引用的文件与实际代码上下文不匹配：" + item["file"])
        existing = _schema_columns(flow, data_results)
        parsed, dynamic = _column_references(tree, flow)
        context_unknown += dynamic
        declared = set(item.get("referenced_columns", []))
        model = set(flow.get("model_features", []))
        model_scoped = item.get("scope") == "model_input" or item.get("problem") == "Missing Value Handling Risk"
        if declared:
            if not existing:
                context_unknown.append("无法确认当前数据集的字段模式")
            elif declared - existing:
                failures.append("引用不存在字段：" + str(sorted(declared - existing)))
        schemas = deepcopy(flow.get("variable_columns", {}))
        for source in flow.get("source_data", []):
            if source.get("variable") and existing:
                schemas.setdefault(source["variable"], list(existing))
        # train_test_split preserves each input's columns in its two outputs.
        # Never fall back from a model subset to the entire uploaded CSV schema.
        for call in flow.get("split", {}).get("calls", []):
            inputs, outputs = call.get("inputs", []), call.get("outputs", [])
            if len(outputs) == 2 * len(inputs):
                for index, source in enumerate(inputs):
                    if source in schemas:
                        for output in outputs[2 * index:2 * index + 2]:
                            schemas.setdefault(output, list(schemas[source]))
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                target, value = statement.targets[0].id, statement.value
                if isinstance(value, ast.Name) and value.id in schemas:
                    schemas[target] = list(schemas[value.id])
                elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
                    if isinstance(value.slice, (ast.List, ast.Tuple)) and all(isinstance(n, ast.Constant) and isinstance(n.value, str) for n in value.slice.elts):
                        schemas[target] = [n.value for n in value.slice.elts]
        for receiver, columns in parsed.items():
            receiver_schema = schemas.get(receiver)
            if receiver_schema and receiver_schema[0] == "__all_except__":
                receiver_schema = list(existing - set(receiver_schema[1:])) if existing else None
            known = set(receiver_schema) if receiver_schema is not None else set()
            if not known:
                context_unknown.append(receiver + " 的字段模式无法确认")
            elif columns - known:
                failures.append("引用不可用字段：" + str(sorted(columns - known)))
            declared.update(columns)
        if model_scoped:
            if not model:
                context_unknown.append("实际模型输入字段无法确认")
            elif declared - model:
                failures.append("超出实际模型输入字段：" + str(sorted(declared - model)))
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
            if call.func.id not in set(dir(builtins)) and call.func.id not in {"GridSearchCV"}:
                unverified.append(call.func.id + "() 的调用签名和兼容性尚未验证")
        context_applied, context_errors, context_detail = _contextual_patch_validation(flow, item, before, after)
        failures += context_errors
        replacement, position_detail = _source_replacement(flow, item, before, after)
        if (replacement is None and flow.get("code_units") and item.get("line")
                and item.get("file") == flow.get("file")):
            failures.append(position_detail)
        if failures:
            status, detail = "Failed Validation", "；".join(dict.fromkeys(failures))
        elif _requires_coordinated_repair(item, before, after):
            status, detail = "Recommendation Only", "该修复涉及拟合/划分、样本或上下游流程，单处替换不能证明完整修复；请协调多处修改并运行对照。"
        elif unverified:
            status, detail = "API Unverified", "语法检查通过，但存在未确认的上下文/API：" + "；".join(dict.fromkeys(unverified))
        elif context_unknown or not replacement or not context_applied:
            status, detail = "Syntax Valid Only", "仅完成片段语法检查，未确认完整上下文；" + position_detail + "；" + "；".join(context_unknown)
        else:
            # Nested scopes and dynamic control flow need a separate lexical
            # proof. Refuse to promote a whole-file AST parse into that proof.
            source_tree = ast.parse(replacement["before_source"])
            line = item["line"]
            nested = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try, ast.With))
                         and n.lineno <= line <= getattr(n, "end_lineno", n.lineno) for n in ast.walk(source_tree))
            if nested or any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)) for n in ast.walk(tree)):
                status, detail = "Recommendation Only", "涉及嵌套作用域或条件/循环路径，无法确认当前位置与下游语义安全。"
            else:
                status, detail = "Statically Validated", context_detail + "；" + position_detail + " 未执行代码，不代表运行成功。"
                replacement["eligible"] = replacement["scope"] == "python_file"
                if not replacement["project_applicable"]:
                    detail += " 仅验证了 Notebook 代码单元，没有原始 Notebook JSON 快照，不导出项目级补丁。"
                item["patch_validation"] = replacement
                item["after_code"] = item["corrected_code"] = str(after)
        item["validation_checks"] = {"syntax": True, "context_applied": bool(replacement and context_applied),
                                     "api_checked": checked, "runtime_executed": False}
        validated.append(_finish_repair(item, status, detail))
    return validated
