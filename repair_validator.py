"""Conservative syntax, context, and known-API validation for repair snippets."""

from __future__ import annotations

import ast
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
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
    callable_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
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
    return loaded_names - imported - callable_names, columns


def _defined_names(tree):
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param))
    }


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
    target_index = next(
        (index for index in candidates if str(before) in str(units[index].get("code", ""))),
        None,
    )
    if target_index is None:
        return False, [], "无法在建议位置逐字定位原始代码，未应用 Patch。"

    original_source = str(units[target_index].get("code", ""))
    units[target_index]["code"] = original_source.replace(str(before), str(after), 1)
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

    before_tree = ast.parse(textwrap.dedent(str(before)))
    after_tree = ast.parse(textwrap.dedent(str(after)))
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
    path = RUNTIME_CLASS_PATHS.get(object_type)
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
        **(suggestion.get("object_types", {})),
    }
    constructor_inputs = {
        **((data_flow or {}).get("constructor_inputs", {})),
        **(suggestion.get("constructor_inputs", {})),
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
            if method in {
                "fit", "transform", "fit_transform", "predict", "fillna", "dropna",
                "mean_difference", "statistical_parity_difference", "disparate_impact",
                "equal_opportunity_difference", "average_odds_difference",
            }:
                unverified.append(f"{receiver or 'expression'}.{method}() 的对象类型无法确认")
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


def validate_repair_suggestions(suggestions, data_flow=None, data_results=None):
    """Validate repairs in their source context without executing project code."""
    validated = []
    existing_columns = set()
    for entry in data_results or []:
        result = entry.get("result", entry)
        existing_columns.update(result.get("data_types", {}))
    model_features = set((data_flow or {}).get("model_features", []))

    for suggestion in suggestions or []:
        item = deepcopy(suggestion)
        context_flow = _flow_context_for_suggestion(data_flow, item)
        after = item.get("after_code") or item.get("corrected_code")
        if not after:
            item["validation_status"] = "Recommendation Only"
            item["validation_detail"] = "该建议仅提供人工处理方向，没有声称可执行的 Patch。"
            validated.append(item)
            continue
        referenced = set(item.get("referenced_columns", []))
        missing = referenced - existing_columns if existing_columns else set()
        outside_model = referenced - model_features if item.get("scope") == "model_input" and model_features else set()
        if missing or outside_model:
            item["validation_status"] = "Failed Validation"
            item["validation_detail"] = (
                f"引用不存在字段：{sorted(missing)}；超出当前模型字段：{sorted(outside_model)}。"
            )
            item["after_code"] = None
            item["corrected_code"] = None
            validated.append(item)
            continue
        try:
            tree = ast.parse(textwrap.dedent(str(after)))
        except SyntaxError as exc:
            item["validation_status"] = "Failed Validation"
            item["validation_detail"] = f"语法校验失败：line {exc.lineno}: {exc.msg}"
            item["after_code"] = None
            item["corrected_code"] = None
            validated.append(item)
            continue

        external_variables, parsed_columns = _snippet_references(tree)
        allowed_variables = _defined_variables(context_flow, item)
        unknown_variables = external_variables - allowed_variables - {
            "pd", "np", "plt", "sns", "os", "sys", "sklearn", "True", "False", "None",
        }
        if unknown_variables:
            item["validation_status"] = "Failed Validation"
            item["validation_detail"] = (
                "Suggestion references variables that are not defined at the proposed location: "
                f"{sorted(unknown_variables)}."
            )
            item["after_code"] = None
            item["corrected_code"] = None
            validated.append(item)
            continue

        variable_columns = context_flow.get("variable_columns", {})
        parsed_missing = set()
        parsed_outside = set()
        for receiver, columns in parsed_columns.items():
            known = variable_columns.get(receiver, [])
            if known and known[0] == "__all_except__":
                known = [column for column in existing_columns if column not in set(known[1:])]
            if known:
                parsed_missing.update(set(columns) - set(known))
            elif existing_columns:
                parsed_missing.update(set(columns) - existing_columns)
            if item.get("scope") == "model_input" and model_features:
                parsed_outside.update(set(columns) - model_features)
        if parsed_missing or parsed_outside:
            item["validation_status"] = "Failed Validation"
            item["validation_detail"] = (
                f"Suggestion references unavailable fields: {sorted(parsed_missing)}; "
                f"fields outside the model-input scope: {sorted(parsed_outside)}."
            )
            item["after_code"] = None
            item["corrected_code"] = None
            validated.append(item)
            continue

        checked, unverified, failures = _api_validation(tree, context_flow, item)
        grid_checked, grid_unverified, grid_failures = _grid_search_validation(
            tree, context_flow,
        )
        checked.extend(grid_checked)
        unverified.extend(grid_unverified)
        failures.extend(grid_failures)
        before = item.get("before_code") or item.get("original_code")
        context_applied, context_errors, context_detail = _contextual_patch_validation(
            context_flow, item, before, after,
        )
        failures.extend(context_errors)
        if failures:
            item["validation_status"] = "Failed Validation"
            item["validation_detail"] = "；".join(failures)
            item["after_code"] = None
            item["corrected_code"] = None
        elif unverified:
            item["validation_status"] = "API Unverified"
            item["validation_detail"] = (
                "AST、变量和字段检查通过；" + context_detail + "；" + "；".join(unverified)
            )
        elif context_applied:
            item["validation_status"] = "Statically Validated"
            item["validation_detail"] = (
                context_detail
                + (" 已知 API 校验：" + "、".join(checked) if checked else "")
            )
        else:
            item["validation_status"] = "Syntax Valid Only"
            item["validation_detail"] = (
                "建议代码通过 AST、变量和字段片段检查，但未能应用到完整上下文；"
                + context_detail
                + (" 已知 API 校验：" + "、".join(checked) if checked else "")
            )
        validated.append(item)
    return validated
