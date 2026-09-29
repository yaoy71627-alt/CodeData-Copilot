"""Conservative syntax, context, and known-API validation for repair snippets."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import importlib.util
import textwrap


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


def validate_repair_suggestions(suggestions, data_flow=None, data_results=None):
    """Attach explicit validation levels without executing user project code."""
    validated = []
    existing_columns = set()
    for entry in data_results or []:
        result = entry.get("result", entry)
        existing_columns.update(result.get("data_types", {}))
    model_features = set((data_flow or {}).get("model_features", []))

    for suggestion in suggestions or []:
        item = deepcopy(suggestion)
        after = item.get("after_code") or item.get("corrected_code")
        if not after:
            item["validation_status"] = "API Unverified"
            item["validation_detail"] = "该建议仅提供人工处理方向，没有可静态校验的 Patch。"
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

        checked, unverified, failures = _api_validation(tree, data_flow, item)
        if failures:
            item["validation_status"] = "Failed Validation"
            item["validation_detail"] = "；".join(failures)
            item["after_code"] = None
            item["corrected_code"] = None
        elif unverified:
            item["validation_status"] = "API Unverified"
            item["validation_detail"] = "AST 语法有效；" + "；".join(unverified)
        elif checked:
            item["validation_status"] = "Fully Validated"
            item["validation_detail"] = (
                "AST 语法、字段约束和已知 API 方法归属均通过静态校验：" + "、".join(checked)
            )
        else:
            item["validation_status"] = "Syntax Valid"
            item["validation_detail"] = "建议代码已通过 AST 语法校验；未发现可由当前 registry 进一步确认的 API 调用。"
        validated.append(item)
    return validated
