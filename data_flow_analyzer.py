"""Lightweight, evidence-based data flow analysis for data science code."""

from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path

import nbformat
import pandas as pd

from notebook_syntax import preprocess_notebook_source


CLASSIFICATION_METRICS = {
    "accuracy_score": "accuracy",
    "precision_score": "precision",
    "recall_score": "recall",
    "f1_score": "f1",
    "roc_auc_score": "roc_auc",
    "average_precision_score": "pr_auc",
    "log_loss": "log_loss",
    "confusion_matrix": "confusion_matrix",
    "classification_report": "classification_report",
}
REGRESSION_METRICS = {
    "mean_absolute_error": "mae",
    "mean_squared_error": "mse",
    "root_mean_squared_error": "rmse",
    "r2_score": "r2",
    "mean_absolute_percentage_error": "mape",
}
PREPROCESSOR_NAMES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
    "SimpleImputer", "KNNImputer", "IterativeImputer", "PCA", "TruncatedSVD",
    "SelectKBest", "SelectPercentile", "VarianceThreshold", "RFE", "RFECV",
    "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "LabelEncoder",
    "ColumnTransformer", "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler",
}
FAIRNESS_OBJECTS = {
    "BinaryLabelDatasetMetric": "Dataset fairness metric",
    "ClassificationMetric": "Prediction fairness metric",
}
FAIRNESS_METHODS = {
    "BinaryLabelDatasetMetric": {
        "mean_difference", "statistical_parity_difference", "disparate_impact",
    },
    "ClassificationMetric": {
        "equal_opportunity_difference", "average_odds_difference",
        "average_abs_odds_difference", "true_positive_rate_difference",
        "mean_difference", "statistical_parity_difference", "disparate_impact",
    },
}

SAFE_DTYPE_NAMES = {
    "str", "string", "object", "category", "bool", "boolean",
    "int", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float", "float16", "float32", "float64", "datetime64[ns]",
}
OBSERVATION_CALLS = {
    "len", "print", "info", "describe", "count", "nunique", "value_counts",
    "memory_usage", "isna", "notna", "sum", "mean", "median", "min", "max",
    "std", "var", "quantile",
}
OBSERVATION_ATTRIBUTES = {"shape", "size", "ndim", "empty", "columns", "dtypes", "index"}


def _role_from_name(name):
    value = str(name or "").casefold()
    if any(marker in value for marker in ("validation", "valid", "_val", "val_")):
        return "validation"
    if "test" in value:
        return "test"
    if "train" in value:
        return "train"
    return None


def _loop_selection_details(node, parents):
    current = (parents or {}).get(node)
    loop = None
    while current is not None:
        if isinstance(current, ast.For):
            loop = current
            break
        current = (parents or {}).get(current)
    if loop is None:
        return {"selection_type": "unknown", "candidate_models": []}
    candidate_models = []
    if isinstance(loop.iter, (ast.List, ast.Tuple)):
        candidate_models = [
            _call_name(item).split(".")[-1]
            for item in loop.iter.elts if isinstance(item, ast.Call)
        ]
        candidate_models = list(dict.fromkeys(filter(None, candidate_models)))
        if len(candidate_models) >= 2:
            return {
                "selection_type": "model_comparison",
                "candidate_models": candidate_models,
            }
    target_names = {
        item.id for item in ast.walk(loop.target) if isinstance(item, ast.Name)
    }
    parameter_markers = {
        "c", "alpha", "lambda", "n_estimators", "max_depth", "min_samples_split",
        "learning_rate", "hidden_layer_sizes", "k", "degree", "gamma",
    }
    if any(name.casefold() in parameter_markers for name in target_names):
        return {"selection_type": "hyperparameter_search", "candidate_models": []}
    for call in (item for item in ast.walk(loop) if isinstance(item, ast.Call)):
        if call is node:
            continue
        constructor = _call_name(call).split(".")[-1]
        argument_names = {
            item.id
            for argument in [*call.args, *(keyword.value for keyword in call.keywords)]
            for item in ast.walk(argument)
            if isinstance(item, ast.Name)
        }
        if target_names & argument_names and constructor.endswith(("Classifier", "Regressor")):
            return {
                "selection_type": "hyperparameter_search",
                "candidate_models": [constructor],
            }
    return {"selection_type": "repeated_test_evaluation", "candidate_models": []}


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
    if isinstance(value, ast.Attribute):
        return _call_name(value)
    return None


def _expr_label(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node)
    if isinstance(node, ast.Subscript):
        base = _expr_label(node.value)
        try:
            suffix = ast.unparse(node.slice)
        except Exception:
            suffix = "..."
        return f"{base}[{suffix}]" if base else None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _assignment_names(node):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    names = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def _literal_value(node, literals):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return literals.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_literal_value(item, literals) for item in node.elts]
        return values if all(value is not None for value in values) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_value(node.left, literals)
        right = _literal_value(node.right, literals)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    return None


def _keyword(call, name):
    return next((item.value for item in call.keywords if item.arg == name), None)


def _constant_keyword(call, name, literals):
    node = _keyword(call, name)
    return _literal_value(node, literals) if node is not None else None


def _safe_dtype_spec(node, literals):
    """Return a JSON-safe pandas dtype spec without evaluating user code."""
    if node is None:
        return None
    literal = _literal_value(node, literals)
    if isinstance(literal, str) and literal.casefold() in SAFE_DTYPE_NAMES:
        return literal
    if isinstance(node, ast.Name) and node.id.casefold() in SAFE_DTYPE_NAMES:
        return node.id.casefold()
    if isinstance(node, ast.Attribute):
        name = _call_name(node).split(".")[-1].casefold()
        if name in SAFE_DTYPE_NAMES:
            return name
    if isinstance(node, ast.Dict):
        output = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = _literal_value(key_node, literals)
            value = _safe_dtype_spec(value_node, literals)
            if not isinstance(key, str) or value is None:
                return None
            output[key] = value
        return output
    return None


def _slice_spec(node, literals):
    """Return a JSON-safe, statically reproducible slice description."""
    if not isinstance(node, ast.Slice):
        return None
    values = {
        "start": _literal_value(node.lower, literals) if node.lower is not None else None,
        "stop": _literal_value(node.upper, literals) if node.upper is not None else None,
        "step": _literal_value(node.step, literals) if node.step is not None else None,
    }
    unresolved = any(
        expression is not None and values[key] is None
        for key, expression in (("start", node.lower), ("stop", node.upper), ("step", node.step))
    )
    return {**values, "reconstructable": not unresolved}


def _selector_spec(node, literals):
    if isinstance(node, ast.Slice):
        return {"kind": "slice", **_slice_spec(node, literals)}
    value = _literal_value(node, literals)
    if isinstance(value, (int, str)):
        return {"kind": "scalar", "value": value, "reconstructable": True}
    if isinstance(value, list):
        return {"kind": "list", "value": value, "reconstructable": True}
    return {"kind": "unknown", "value": _expr_label(node), "reconstructable": False}


def _dataframe_selection_spec(node, literals):
    """Describe an allowlisted dataframe selection without evaluating code."""
    copied = False
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
        and not node.args
        and not node.keywords
    ):
        copied = True
        node = node.func.value
    if isinstance(node, ast.Name) and copied:
        return {"operation": "assign", "input": node.id}
    if not isinstance(node, ast.Subscript):
        return None
    if (
        isinstance(node.value, ast.Attribute)
        and node.value.attr in {"loc", "iloc"}
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
    ):
        return {
            "operation": f"{node.value.attr}_selection",
            "input": _expr_label(node.value.value),
            "row_selector": _selector_spec(node.slice.elts[0], literals),
            "column_selector": _selector_spec(node.slice.elts[1], literals),
        }
    columns = _literal_value(node.slice, literals)
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(columns, list) and all(isinstance(item, str) for item in columns):
        return {
            "operation": "column_selection",
            "input": _expr_label(node.value),
            "columns": list(columns),
        }
    return None


def _unwrap_copy_call(node, literals):
    """Remove outer ``DataFrame.copy`` calls that cannot change row lineage.

    Only the pandas-compatible ``deep`` boolean is accepted.  The helper does
    not evaluate the expression; it merely exposes the inner AST node so a
    chained allowlisted operation such as ``df.dropna().copy()`` can be
    recorded as ``dropna``.
    """
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
    ):
        if node.args:
            return node
        unsupported = [keyword for keyword in node.keywords if keyword.arg != "deep"]
        deep_node = _keyword(node, "deep")
        deep = _literal_value(deep_node, literals) if deep_node is not None else True
        if unsupported or not isinstance(deep, bool):
            return node
        node = node.func.value
    return node


def _event_location(node, file_name=None, cell=None):
    return {
        "file": file_name,
        "cell": cell,
        "line": getattr(node, "lineno", None),
        "end_line": getattr(node, "end_lineno", None),
    }


def _empty_flow(file_name=None):
    return {
        "file": file_name,
        "source_data": [],
        "feature_matrix": None,
        "target": None,
        "selected_features": [],
        "excluded_features": [],
        "model_features": [],
        "split": {
            "detected": False,
            "method": None,
            "line": None,
            "cell": None,
            "input_features": None,
            "input_target": None,
            "train": [],
            "test": [],
            "test_size": None,
            "random_state": None,
            "stratify": None,
            "train_detected": False,
            "validation_detected": False,
            "test_detected": False,
            "split_roles": {"train": [], "validation": [], "test": []},
            "calls": [],
            "slices": [],
        },
        "preprocessing": [],
        "models": [],
        "fit": [],
        "predict": [],
        "metrics": [],
        "evaluation_detected": False,
        "task_type": {"type": "Unknown", "confidence": "Low", "evidence": []},
        "metric_coverage": {"status": "Unable to verify", "detected": [], "suggested": []},
        "split_role_analysis": {
            "train_detected": False,
            "validation_detected": False,
            "test_detected": False,
            "split_roles": {"train": [], "validation": [], "test": []},
            "risks": [],
        },
        "fairness": {
            "evaluation_exists": False,
            "dataset_metrics": [],
            "prediction_metrics": [],
            "mitigations": [],
            "before_after_comparison": False,
            "coverage": "No fairness evaluation detected",
        },
        "object_types": {},
        "constructor_inputs": {},
        "possible_object_types": {},
        "operations": [],
        "replay_steps": [],
        "definitions": {},
        "reconstruction_limits": [],
        "variable_columns": {},
        "lineage": [],
        "code_units": [],
        "notebook": {"execution_order": [], "hidden_state_risk": False},
    }


class _FlowBuilder:
    def __init__(self, file_name=None):
        self.flow = _empty_flow(file_name)
        self.literals = {}
        self.constructors = {}
        self.variable_columns = {}
        self.object_types = {}
        self.constructor_inputs = {}
        self.constructor_options = {}
        self.possible_object_types = {}
        self.role_usage = {"train": [], "validation": [], "test": []}
        self.sequence = 0

    def _add_replay_step(self, operation, node, cell, **details):
        step = {
            "order": len(self.flow["replay_steps"]) + 1,
            "operation": operation,
            **details,
            **_event_location(node, self.flow["file"], cell),
        }
        self.flow["replay_steps"].append(step)
        return step

    def _record_assignment(self, node, cell):
        names = _assignment_names(node)
        if not names:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                ):
                    continue
                column = _literal_value(target.slice, self.literals)
                if not isinstance(column, str):
                    continue
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "astype"
                ):
                    selection = _dataframe_selection_spec(value.func.value, self.literals)
                    dtype_node = value.args[0] if value.args else _keyword(value, "dtype")
                    dtype = _safe_dtype_spec(dtype_node, self.literals)
                    if (
                        selection
                        and selection.get("operation") == "column_selection"
                        and selection.get("input") == target.value.id
                        and len(selection.get("columns") or []) == 1
                    ):
                        self._add_replay_step(
                            "astype_column", node, cell,
                            input=target.value.id, output=target.value.id,
                            source_column=selection["columns"][0], column=column,
                            dtype=dtype, reconstructable=dtype is not None,
                            expression=_expr_label(value),
                        )
                        continue
                literal = _literal_value(node.value, self.literals)
                self._add_replay_step(
                    "set_column", node, cell, input=target.value.id, output=target.value.id,
                    column=column, value=literal,
                    reconstructable=literal is not None,
                    expression=_expr_label(node.value),
                )
            return
        replay_count_before = len(self.flow["replay_steps"])
        for name in names:
            self.flow["definitions"].setdefault(name, []).append(
                _event_location(node, self.flow["file"], cell)
            )
        value = node.value
        literal = _literal_value(value, self.literals)
        if len(names) == 1 and literal is not None:
            self.literals[names[0]] = literal

        # ``copy`` is a no-op for row/value lineage.  Unwrapping it here lets
        # the normal allowlist record the meaningful inner chained operation.
        if len(names) == 1:
            value = _unwrap_copy_call(value, self.literals)

        if len(names) == 1:
            observation_source = None
            if (
                isinstance(value, ast.Call)
                and _call_name(value).split(".")[-1] in OBSERVATION_CALLS
            ):
                observation_source = next((
                    candidate.id for candidate in ast.walk(value)
                    if isinstance(candidate, ast.Name)
                    and (
                        candidate.id in self.variable_columns
                        or self.object_types.get(candidate.id) in {"DataFrame", "Series"}
                        or any(
                            source.get("variable") == candidate.id
                            for source in self.flow.get("source_data", [])
                        )
                    )
                ), None)
            metadata_node = value.value if isinstance(value, ast.Subscript) else value
            if (
                observation_source is None
                and isinstance(metadata_node, ast.Attribute)
                and metadata_node.attr in OBSERVATION_ATTRIBUTES
            ):
                candidate = _expr_label(metadata_node.value)
                if (
                    candidate in self.variable_columns
                    or self.object_types.get(candidate) in {"DataFrame", "Series"}
                ):
                    observation_source = candidate
            if observation_source:
                self._add_replay_step(
                    "observe", node, cell, input=observation_source, output=names[0],
                    expression=_expr_label(value),
                )

        selection = _dataframe_selection_spec(value, self.literals) if len(names) == 1 else None
        if selection:
            name = names[0]
            operation = selection["operation"]
            source = selection.get("input")
            if operation == "assign":
                if source in self.variable_columns:
                    self.variable_columns[name] = list(self.variable_columns[source])
                if source in self.object_types:
                    self.object_types[name] = self.object_types[source]
                self._add_replay_step("assign", node, cell, input=source, output=name)
            elif operation == "column_selection":
                columns = list(selection.get("columns") or [])
                self.variable_columns[name] = columns
                self.object_types[name] = (
                    "Series" if len(columns) == 1 and name.casefold() in {"y", "target", "label"}
                    else "DataFrame"
                )
                if name.casefold() in {"y", "target", "label"} and len(columns) == 1:
                    self.flow["target"] = {"variable": name, "column": columns[0], "source": source}
                elif name.casefold().startswith("x"):
                    self.flow["feature_matrix"] = name
                    self.flow["selected_features"] = columns
                self.flow["lineage"].append({
                    "from": source, "to": name, "operation": operation, "columns": columns,
                })
                self._add_replay_step(
                    operation, node, cell, input=source, output=name, columns=columns,
                )
            else:
                column_selector = selection.get("column_selector") or {}
                selected_columns = (
                    list(column_selector.get("value"))
                    if column_selector.get("kind") == "list"
                    and isinstance(column_selector.get("value"), list)
                    else None
                )
                if selected_columns and all(isinstance(item, str) for item in selected_columns):
                    self.variable_columns[name] = selected_columns
                self.object_types[name] = "DataFrame"
                if name.casefold().startswith("x"):
                    self.flow["feature_matrix"] = name
                    if selected_columns:
                        self.flow["selected_features"] = selected_columns
                if name.casefold() in {"y", "target", "label"}:
                    target_column = (
                        column_selector.get("value")
                        if column_selector.get("kind") == "scalar"
                        and isinstance(column_selector.get("value"), str)
                        else None
                    )
                    self.flow["target"] = {
                        "variable": name, "column": target_column, "source": source,
                    }
                self.flow["lineage"].append({
                    "from": source, "to": name, "operation": operation,
                })
                self._add_replay_step(
                    operation, node, cell, input=source, output=name,
                    row_selector=selection.get("row_selector"),
                    column_selector=column_selector,
                )

        if isinstance(value, ast.Call):
            call_name = _call_name(value)
            short_name = call_name.split(".")[-1]
            if len(names) == 1 and short_name not in {
                "copy", "fit_transform", "transform", "fit_resample",
                *OBSERVATION_CALLS,
            }:
                self.constructors[names[0]] = short_name
                self.object_types[names[0]] = short_name
                self.constructor_inputs[names[0]] = {
                    "positional": [_expr_label(arg) for arg in value.args],
                    "keywords": {item.arg: _expr_label(item.value) for item in value.keywords if item.arg},
                }
                self.constructor_options[names[0]] = {
                    "positional": [
                        _literal_value(argument, self.literals) for argument in value.args
                    ],
                    "positional_reconstructable": all(
                        _literal_value(argument, self.literals) is not None
                        for argument in value.args
                    ),
                    "keywords": {
                        item.arg: _literal_value(item.value, self.literals)
                        for item in value.keywords if item.arg
                    },
                    "keyword_reconstructable": {
                        item.arg: _literal_value(item.value, self.literals) is not None
                        for item in value.keywords if item.arg
                    },
                }
            if short_name.startswith("read_") or call_name.startswith("pd.read_"):
                source = {
                    "variable": names[0],
                    "reader": call_name,
                    "path": _literal_value(value.args[0], self.literals) if value.args else None,
                    **_event_location(node, self.flow["file"], cell),
                }
                self.flow["source_data"].append(source)
                self.object_types[names[0]] = "DataFrame"
                self.flow["lineage"].append({"from": source.get("path") or call_name, "to": names[0], "operation": "load"})
                self._add_replay_step(
                    "load", node, cell, output=names[0], path=source.get("path"), reader=call_name,
                )

            if short_name == "astype" and len(names) == 1 and isinstance(value.func, ast.Attribute):
                receiver_expression = value.func.value
                receiver_selection = _dataframe_selection_spec(
                    receiver_expression, self.literals,
                )
                input_name = _expr_label(receiver_expression)
                input_columns = None
                scalar_column_selection = False
                if (
                    receiver_selection
                    and receiver_selection.get("operation") == "column_selection"
                ):
                    input_name = receiver_selection.get("input")
                    input_columns = list(receiver_selection.get("columns") or [])
                    scalar_column_selection = (
                        isinstance(receiver_expression, ast.Subscript)
                        and isinstance(
                            _literal_value(receiver_expression.slice, self.literals),
                            str,
                        )
                    )
                dtype_node = value.args[0] if value.args else _keyword(value, "dtype")
                dtype = _safe_dtype_spec(dtype_node, self.literals)
                self._add_replay_step(
                    "astype", node, cell, input=input_name, output=names[0],
                    input_columns=input_columns, dtype=dtype,
                    reconstructable=dtype is not None,
                    expression=_expr_label(value),
                )
                if input_columns:
                    self.variable_columns[names[0]] = input_columns
                elif input_name in self.variable_columns:
                    self.variable_columns[names[0]] = list(self.variable_columns[input_name])
                output_columns = self.variable_columns.get(names[0], [])
                input_is_series = self.object_types.get(input_name) == "Series"
                self.object_types[names[0]] = (
                    "Series"
                    if scalar_column_selection or input_is_series
                    else self.object_types.get(input_name, "DataFrame")
                )
                if (
                    names[0].casefold() in {"y", "target", "label"}
                    and len(output_columns) == 1
                ):
                    self.flow["target"] = {
                        "variable": names[0],
                        "column": output_columns[0],
                        "source": input_name,
                    }
                self.flow["lineage"].append({
                    "from": input_name, "to": names[0], "operation": "astype",
                    "columns": list(output_columns),
                })

            if short_name == "get_dummies" and len(names) == 1 and value.args:
                input_expression = value.args[0]
                input_selection = _dataframe_selection_spec(input_expression, self.literals)
                input_name = _expr_label(input_expression)
                input_columns = None
                if input_selection and input_selection.get("operation") == "column_selection":
                    input_name = input_selection.get("input")
                    input_columns = list(input_selection.get("columns") or [])
                columns_node = _keyword(value, "columns")
                drop_first_node = _keyword(value, "drop_first")
                dummy_na_node = _keyword(value, "dummy_na")
                prefix_node = _keyword(value, "prefix")
                prefix_sep_node = _keyword(value, "prefix_sep")
                sparse_node = _keyword(value, "sparse")
                dtype_node = _keyword(value, "dtype")
                columns = _constant_keyword(value, "columns", self.literals)
                drop_first = _constant_keyword(value, "drop_first", self.literals)
                dummy_na = _constant_keyword(value, "dummy_na", self.literals)
                prefix = _constant_keyword(value, "prefix", self.literals)
                prefix_sep = _constant_keyword(value, "prefix_sep", self.literals)
                sparse = _constant_keyword(value, "sparse", self.literals)
                dtype = _safe_dtype_spec(dtype_node, self.literals)
                supported_keywords = {
                    "prefix", "prefix_sep", "dummy_na", "columns", "sparse",
                    "drop_first", "dtype",
                }
                unknown_keywords = [
                    item.arg for item in value.keywords
                    if item.arg not in supported_keywords
                ]
                self.object_types[names[0]] = "DataFrame"
                if input_columns:
                    self.variable_columns[names[0]] = input_columns
                elif input_name in self.variable_columns:
                    self.variable_columns[names[0]] = list(self.variable_columns[input_name])
                if names[0].casefold().startswith("x"):
                    self.flow["feature_matrix"] = names[0]
                    if names[0] in self.variable_columns:
                        self.flow["selected_features"] = list(
                            self.variable_columns[names[0]]
                        )
                self._add_replay_step(
                    "get_dummies", node, cell, input=input_name, output=names[0],
                    input_columns=input_columns, columns=columns,
                    drop_first=bool(drop_first), dummy_na=bool(dummy_na),
                    prefix=prefix, prefix_sep=prefix_sep, sparse=sparse, dtype=dtype,
                    reconstructable=(
                        len(value.args) == 1
                        and (columns_node is None or columns is not None)
                        and (drop_first_node is None or drop_first is not None)
                        and (dummy_na_node is None or dummy_na is not None)
                        and (prefix_node is None or prefix is not None)
                        and (prefix_sep_node is None or prefix_sep is not None)
                        and (sparse_node is None or sparse is not None)
                        and (dtype_node is None or dtype is not None)
                        and not unknown_keywords
                    ),
                )
                self.flow["lineage"].append({
                    "from": input_name, "to": names[0], "operation": "get_dummies",
                    "source_columns": list(input_columns or self.variable_columns.get(input_name, [])),
                })

            if (
                short_name in {"fit_transform", "transform"}
                and len(names) == 1
                and isinstance(value.func, ast.Attribute)
                and value.args
            ):
                transformer = _receiver_name(value)
                transformer_type = self.constructors.get(transformer)
                input_name = _expr_label(value.args[0])
                input_columns = list(self.variable_columns.get(input_name, []))
                options = deepcopy(self.constructor_options.get(transformer, {}))
                exactly_supported = (
                    transformer_type == "StandardScaler"
                    and short_name == "fit_transform"
                )
                self._add_replay_step(
                    "preprocess_transform", node, cell,
                    input=input_name, output=names[0], transformer=transformer,
                    transformer_type=transformer_type, method=short_name,
                    options=options, row_preserving=transformer_type in PREPROCESSOR_NAMES,
                    value_reconstructable=exactly_supported,
                    expression=_expr_label(value),
                )
                if input_columns:
                    self.variable_columns[names[0]] = input_columns
                self.object_types[names[0]] = "DataFrame"
                if names[0].casefold().startswith("x"):
                    self.flow["feature_matrix"] = names[0]
                    if input_columns:
                        self.flow["selected_features"] = input_columns
                self.flow["lineage"].append({
                    "from": input_name, "to": names[0],
                    "operation": short_name, "transformer": transformer_type,
                    "value_reconstruction": (
                        "Exact" if exactly_supported else "Partial"
                    ),
                })

            if short_name == "train_test_split" and len(names) >= 2:
                args = [_expr_label(arg) for arg in value.args]
                test_size_node = _keyword(value, "test_size")
                train_size_node = _keyword(value, "train_size")
                test_size = _constant_keyword(value, "test_size", self.literals)
                train = [name for name in names if "train" in name.casefold()]
                validation = [name for name in names if _role_from_name(name) == "validation"]
                test = [name for name in names if _role_from_name(name) == "test"]
                split = self.flow["split"]
                role_variables = {
                    "train": train,
                    "validation": validation,
                    "test": test,
                }
                for role, variables in role_variables.items():
                    split["split_roles"][role] = list(dict.fromkeys([
                        *split["split_roles"].get(role, []), *variables,
                    ]))
                    split[f"{role}_detected"] = bool(split["split_roles"][role])
                split_event = {
                    "line": node.lineno,
                    "cell": cell,
                    "inputs": args,
                    "outputs": names,
                    "roles": role_variables,
                    "test_size": test_size,
                    "test_size_provided": test_size_node is not None,
                    "test_size_reconstructable": test_size_node is None or test_size is not None,
                    "train_size_provided": train_size_node is not None,
                    "random_state": _constant_keyword(value, "random_state", self.literals),
                }
                split["calls"].append(split_event)
                split.update({
                    "detected": True,
                    "method": "train_test_split",
                    "line": node.lineno,
                    "cell": cell,
                    "input_features": args[0] if args else None,
                    "input_target": args[1] if len(args) > 1 else None,
                    "train": train,
                    "test": test,
                    "test_size": test_size,
                    "test_size_provided": test_size_node is not None,
                    "test_size_reconstructable": test_size_node is None or test_size is not None,
                    "train_size_provided": train_size_node is not None,
                    "random_state": _constant_keyword(value, "random_state", self.literals),
                    "stratify": _expr_label(_keyword(value, "stratify")) if _keyword(value, "stratify") else None,
                })
                if args:
                    for output in train + validation + test:
                        self.flow["lineage"].append({
                            "from": args[0] if output.casefold().startswith("x") else (args[1] if len(args) > 1 else None),
                            "to": output,
                            "operation": "train_test_split",
                        })
                self._add_replay_step(
                    "train_test_split", node, cell, inputs=args, outputs=names,
                    roles=role_variables, test_size=split_event["test_size"],
                    test_size_provided=split_event["test_size_provided"],
                    test_size_reconstructable=split_event["test_size_reconstructable"],
                    train_size_provided=split_event["train_size_provided"],
                    random_state=split_event["random_state"],
                    stratify=split.get("stratify"),
                )

            if short_name == "split" and len(names) >= 2:
                role_variables = {
                    role: [name for name in names if _role_from_name(name) == role]
                    for role in ("train", "validation", "test")
                }
                if any(role_variables.values()):
                    receiver = _receiver_name(value)
                    fractions = _literal_value(value.args[0], self.literals) if value.args else None
                    split = self.flow["split"]
                    split["detected"] = True
                    split["method"] = "dataset_split"
                    split["input_features"] = receiver
                    split["line"] = node.lineno
                    split["cell"] = cell
                    for role, variables in role_variables.items():
                        split["split_roles"][role] = list(dict.fromkeys([
                            *split["split_roles"].get(role, []), *variables,
                        ]))
                        split[f"{role}_detected"] = bool(split["split_roles"][role])
                    event = {
                        "line": node.lineno, "cell": cell, "inputs": [receiver],
                        "outputs": names, "roles": role_variables, "fractions": fractions,
                        "shuffle": _constant_keyword(value, "shuffle", self.literals),
                    }
                    split["calls"].append(event)
                    self._add_replay_step(
                        "dataset_split", node, cell, input=receiver, outputs=names,
                        roles=role_variables, fractions=fractions, shuffle=event["shuffle"],
                    )

            if short_name in {"dropna", "drop_duplicates"} and len(names) == 1:
                receiver = _receiver_name(value)
                input_columns = None
                receiver_expression = value.func.value if isinstance(value.func, ast.Attribute) else None
                if isinstance(receiver_expression, ast.Subscript):
                    receiver = _expr_label(receiver_expression.value)
                    input_columns = _literal_value(receiver_expression.slice, self.literals)
                    if isinstance(input_columns, str):
                        input_columns = [input_columns]
                subset = _constant_keyword(value, "subset", self.literals)
                if isinstance(subset, str):
                    subset = [subset]
                self.flow["operations"].append({
                    "type": short_name,
                    "receiver": receiver,
                    "assigned_to": names[0],
                    "subset": subset,
                    "input_columns": input_columns,
                    "inplace": bool(_constant_keyword(value, "inplace", self.literals)),
                    **_event_location(node, self.flow["file"], cell),
                })
                self._add_replay_step(
                    short_name, node, cell, input=receiver, output=names[0],
                    subset=subset, input_columns=input_columns,
                    inplace=bool(_constant_keyword(value, "inplace", self.literals)),
                )
                if receiver in self.variable_columns:
                    self.variable_columns[names[0]] = list(self.variable_columns[receiver])
                elif isinstance(input_columns, list):
                    self.variable_columns[names[0]] = list(input_columns)

            if short_name == "predict" and len(names) == 1:
                prediction_input = _expr_label(value.args[0]) if value.args else None
                event = {
                    "object": _receiver_name(value), "X": prediction_input, "output": names[0],
                    **_event_location(node, self.flow["file"], cell),
                }
                self.flow["predict"].append(event)
                self.flow["lineage"].append({"from": prediction_input, "to": names[0], "operation": "predict"})

        if isinstance(value, ast.Name) and len(names) == 1 and not selection:
            if value.id in self.variable_columns:
                self.variable_columns[names[0]] = list(self.variable_columns[value.id])
            self.flow["lineage"].append({"from": value.id, "to": names[0], "operation": "assign"})
            if value.id in self.object_types:
                self.object_types[names[0]] = self.object_types[value.id]
            self._add_replay_step("assign", node, cell, input=value.id, output=names[0])

        if isinstance(value, ast.Subscript) and len(names) == 1 and not selection:
            source = _expr_label(value.value)
            columns = _literal_value(value.slice, self.literals)
            if isinstance(columns, str):
                columns = [columns]
            if isinstance(columns, list) and all(isinstance(item, str) for item in columns):
                self.variable_columns[names[0]] = list(columns)
                self.object_types[names[0]] = "Series" if len(columns) == 1 and names[0].casefold() in {"y", "target", "label"} else "DataFrame"
                if names[0].casefold() in {"y", "target", "label"} and len(columns) == 1:
                    self.flow["target"] = {"variable": names[0], "column": columns[0], "source": source}
                elif names[0].casefold().startswith("x"):
                    self.flow["feature_matrix"] = names[0]
                    self.flow["selected_features"] = list(columns)
                self.flow["lineage"].append({"from": source, "to": names[0], "operation": "column_selection", "columns": list(columns)})
                self._add_replay_step(
                    "column_selection", node, cell, input=source, output=names[0], columns=list(columns),
                )
            else:
                if (
                    isinstance(value.value, ast.Attribute)
                    and value.value.attr in {"iloc", "loc"}
                    and isinstance(value.slice, ast.Tuple)
                    and len(value.slice.elts) == 2
                ):
                    source = _expr_label(value.value.value)
                    row_selector = _selector_spec(value.slice.elts[0], self.literals)
                    column_selector = _selector_spec(value.slice.elts[1], self.literals)
                    self._add_replay_step(
                        f"{value.value.attr}_selection", node, cell,
                        input=source, output=names[0],
                        row_selector=row_selector, column_selector=column_selector,
                    )
                    if names[0].casefold().startswith("x"):
                        self.flow["feature_matrix"] = names[0]
                    if names[0].casefold() in {"y", "target", "label"}:
                        self.flow["target"] = {
                            "variable": names[0], "column": None, "source": source,
                        }
                    self.flow["lineage"].append({
                        "from": source, "to": names[0],
                        "operation": f"{value.value.attr}_selection",
                    })
                    return
                slice_spec = _slice_spec(value.slice, self.literals)
                base_node = value.value
                if isinstance(base_node, ast.Attribute) and base_node.attr in {"iloc", "loc"}:
                    source = _expr_label(base_node.value)
                if slice_spec and source:
                    role = _role_from_name(names[0])
                    if (
                        role is None
                        and slice_spec.get("reconstructable")
                        and slice_spec.get("start") in {None, 0}
                        and slice_spec.get("stop") is not None
                    ):
                        role = "train"
                    self._add_replay_step(
                        "row_slice", node, cell, input=source, output=names[0],
                        slice=slice_spec, role=role,
                    )
                    self.flow["lineage"].append({
                        "from": source, "to": names[0], "operation": "row_slice",
                        "slice": slice_spec,
                    })
                    if source in self.variable_columns:
                        self.variable_columns[names[0]] = list(self.variable_columns[source])
                    if role:
                        split = self.flow["split"]
                        split["detected"] = True
                        split["method"] = "slice"
                        split["input_features"] = split.get("input_features") or source
                        split["line"] = node.lineno
                        split["cell"] = cell
                        split["split_roles"][role] = list(dict.fromkeys([
                            *split["split_roles"].get(role, []), names[0],
                        ]))
                        split[f"{role}_detected"] = True
                        split["slices"].append({
                            "source": source, "output": names[0], "role": role,
                            "slice": slice_spec, **_event_location(node, self.flow["file"], cell),
                        })
                        if role == "train":
                            split["train"] = list(dict.fromkeys([*split.get("train", []), names[0]]))
                        if role == "test":
                            split["test"] = list(dict.fromkeys([*split.get("test", []), names[0]]))

        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and len(names) == 1:
            if value.func.attr == "drop":
                source = _expr_label(value.func.value)
                columns = _constant_keyword(value, "columns", self.literals)
                if columns is None and value.args:
                    columns = _literal_value(value.args[0], self.literals)
                if isinstance(columns, str):
                    columns = [columns]
                if isinstance(columns, list):
                    excluded_columns = [str(item) for item in columns]
                    if names[0].casefold().startswith("x"):
                        self.flow["feature_matrix"] = names[0]
                        self.flow["excluded_features"] = excluded_columns
                    self.variable_columns[names[0]] = ["__all_except__", *excluded_columns]
                    self.object_types[names[0]] = "DataFrame"
                    self.flow["lineage"].append({"from": source, "to": names[0], "operation": "drop_columns", "columns": list(columns)})
                    self._add_replay_step(
                        "drop_columns", node, cell, input=source, output=names[0], columns=list(columns),
                    )

        if len(self.flow["replay_steps"]) == replay_count_before and len(names) == 1:
            referenced = [
                candidate.id for candidate in ast.walk(value)
                if isinstance(candidate, ast.Name)
                and (
                    candidate.id in self.variable_columns
                    or self.object_types.get(candidate.id) in {"DataFrame", "Series"}
                    or any(
                        source.get("variable") == candidate.id
                        for source in self.flow.get("source_data", [])
                    )
                )
            ]
            if referenced and isinstance(value, (ast.Call, ast.Subscript, ast.BinOp, ast.Compare)):
                self._add_replay_step(
                    "unsupported", node, cell, input=referenced[0], output=names[0],
                    expression=_expr_label(value),
                    reason="dataframe transformation is outside the safe replay allowlist",
                )
                self.flow["reconstruction_limits"].append({
                    "variable": names[0], "expression": _expr_label(value),
                    "reason": "unsupported dataframe transformation",
                    **_event_location(node, self.flow["file"], cell),
                })

    def _record_call(self, call, cell, parents=None):
        self.sequence += 1
        full_name = _call_name(call)
        short_name = full_name.split(".")[-1]
        receiver = _receiver_name(call)
        args = [_expr_label(arg) for arg in call.args]
        argument_names = {
            node.id for argument in call.args for node in ast.walk(argument) if isinstance(node, ast.Name)
        }
        in_loop = False
        current = (parents or {}).get(call)
        while current is not None:
            if isinstance(current, (ast.For, ast.While, ast.comprehension)):
                in_loop = True
                break
            current = (parents or {}).get(current)

        purpose = None
        selection_details = {"selection_type": None, "candidate_models": []}
        if short_name == "fit":
            purpose = "fit"
        elif short_name in {"fit_transform", "fit_resample"}:
            purpose = "transform fit"
        elif short_name == "predict":
            purpose = "predict"
        elif short_name in CLASSIFICATION_METRICS or short_name in REGRESSION_METRICS:
            if in_loop:
                selection_details = _loop_selection_details(call, parents)
                purpose = "test selection"
            else:
                purpose = "metric"
        if purpose:
            for variable in argument_names:
                role = _role_from_name(variable)
                if role:
                    self.role_usage[role].append({
                        "variable": variable,
                        "purpose": purpose,
                        "object": receiver,
                        "in_loop": in_loop,
                        **selection_details,
                        **_event_location(call, self.flow["file"], cell),
                    })

        if short_name in {"fit", "fit_transform", "fit_resample", "transform"}:
            constructor = self.constructors.get(receiver)
            event = {
                "object": receiver,
                "operation": short_name,
                "X": args[0] if args else None,
                "y": args[1] if len(args) > 1 else None,
                **_event_location(call, self.flow["file"], cell),
            }
            if short_name in {"fit", "fit_transform", "fit_resample"}:
                self.flow["fit"].append(event)
            if constructor in PREPROCESSOR_NAMES or short_name in {"fit_transform", "fit_resample", "transform"}:
                self.flow["preprocessing"].append({**event, "kind": constructor or "transformer"})

        if short_name == "predict" and not any(
            item.get("line") == call.lineno and item.get("cell") == cell for item in self.flow["predict"]
        ):
            self.flow["predict"].append({
                "object": receiver, "X": args[0] if args else None, "output": None,
                **_event_location(call, self.flow["file"], cell),
            })

        if short_name in CLASSIFICATION_METRICS or short_name in REGRESSION_METRICS:
            family = "Classification" if short_name in CLASSIFICATION_METRICS else "Regression"
            label = (CLASSIFICATION_METRICS | REGRESSION_METRICS)[short_name]
            self.flow["metrics"].append({
                "name": short_name,
                "label": label,
                "family": family,
                "target": args[0] if args else None,
                "prediction": args[1] if len(args) > 1 else None,
                **_event_location(call, self.flow["file"], cell),
            })

        object_type = self.object_types.get(receiver)
        if object_type in FAIRNESS_METHODS and short_name in FAIRNESS_METHODS[object_type]:
            semantic_scope = FAIRNESS_OBJECTS[object_type]
            fairness_event = {
                "object": receiver,
                "object_type": object_type,
                "method": short_name,
                "semantic_scope": semantic_scope,
                "in_loop": in_loop,
                "sequence": self.sequence,
                **_event_location(call, self.flow["file"], cell),
            }
            key = "dataset_metrics" if object_type == "BinaryLabelDatasetMetric" else "prediction_metrics"
            self.flow["fairness"][key].append(fairness_event)
        if object_type == "Reweighing" and short_name in {"fit", "transform", "fit_transform"}:
            self.flow["fairness"]["mitigations"].append({
                "object": receiver,
                "type": "Reweighing",
                "operation": short_name,
                "sequence": self.sequence,
                **_event_location(call, self.flow["file"], cell),
            })

        if short_name in {"dropna", "drop_duplicates"} and not any(
            item.get("line") == call.lineno and item.get("cell") == cell for item in self.flow["operations"]
        ):
            subset = _constant_keyword(call, "subset", self.literals)
            if isinstance(subset, str):
                subset = [subset]
            self.flow["operations"].append({
                "type": short_name,
                "receiver": receiver,
                "assigned_to": receiver if _constant_keyword(call, "inplace", self.literals) else None,
                "subset": subset,
                "inplace": bool(_constant_keyword(call, "inplace", self.literals)),
                **_event_location(call, self.flow["file"], cell),
            })
            self._add_replay_step(
                short_name, call, cell, input=receiver,
                output=receiver if _constant_keyword(call, "inplace", self.literals) else None,
                subset=subset, inplace=bool(_constant_keyword(call, "inplace", self.literals)),
            )

        if short_name == "drop" and bool(_constant_keyword(call, "inplace", self.literals)):
            already_recorded = any(
                step.get("line") == call.lineno and step.get("cell") == cell
                and step.get("operation") == "drop_columns"
                for step in self.flow["replay_steps"]
            )
            columns = _constant_keyword(call, "columns", self.literals)
            axis = _constant_keyword(call, "axis", self.literals)
            if columns is None and axis in {1, "columns"} and call.args:
                columns = _literal_value(call.args[0], self.literals)
            if isinstance(columns, str):
                columns = [columns]
            if not already_recorded and receiver and isinstance(columns, list):
                self._add_replay_step(
                    "drop_columns", call, cell, input=receiver, output=receiver,
                    columns=list(columns), inplace=True,
                )
                existing = self.variable_columns.get(receiver)
                if existing and existing[0] != "__all_except__":
                    self.variable_columns[receiver] = [
                        column for column in existing if column not in set(columns)
                    ]

    def process(self, code, cell=None):
        self.flow["code_units"].append({"cell": cell, "code": code})
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        for loop in (item for item in ast.walk(tree) if isinstance(item, ast.For)):
            if not isinstance(loop.target, ast.Name) or not isinstance(loop.iter, (ast.List, ast.Tuple)):
                continue
            candidates = []
            for element in loop.iter.elts:
                if isinstance(element, ast.Call):
                    candidate = _call_name(element).split(".")[-1]
                    if candidate:
                        candidates.append(candidate)
            if candidates:
                self.possible_object_types[loop.target.id] = list(dict.fromkeys(candidates))
                self.flow["definitions"].setdefault(loop.target.id, []).append(
                    _event_location(loop, self.flow["file"], cell)
                )
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        # Process assignments and calls in source order.  Assignment gets
        # priority on the same line so constructor/receiver metadata exists
        # before the nested call is classified; duplicate row operations are
        # suppressed by their original location.
        nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Call))
        ]
        for node in sorted(
            nodes,
            key=lambda item: (
                item.lineno, item.col_offset,
                0 if isinstance(item, (ast.Assign, ast.AnnAssign)) else 1,
            ),
        ):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._record_assignment(node, cell)
            else:
                self._record_call(node, cell, parents)

    def finish(self):
        flow = self.flow
        flow["variable_columns"] = deepcopy(self.variable_columns)
        flow["object_types"] = deepcopy(self.object_types)
        flow["constructor_inputs"] = deepcopy(self.constructor_inputs)
        flow["constructor_options"] = deepcopy(self.constructor_options)
        flow["possible_object_types"] = deepcopy(self.possible_object_types)
        matrix = flow.get("split", {}).get("input_features") or flow.get("feature_matrix")
        if matrix and matrix in self.variable_columns:
            selected = self.variable_columns[matrix]
            flow["feature_matrix"] = matrix
            if selected and selected[0] == "__all_except__":
                flow["excluded_features"] = list(selected[1:])
            else:
                flow["selected_features"] = list(selected)
                flow["model_features"] = list(selected)
        elif flow["selected_features"]:
            flow["model_features"] = list(flow["selected_features"])
        constructors = []
        for variable, constructor in self.constructors.items():
            if constructor in PREPROCESSOR_NAMES:
                continue
            if constructor == "Pipeline" or constructor.endswith(("Classifier", "Regressor")):
                constructors.append({"variable": variable, "type": constructor})
        flow["models"] = constructors
        split = flow["split"]
        if split.get("method") == "slice":
            has_train = bool(split.get("split_roles", {}).get("train"))
            has_holdout = bool(
                split.get("split_roles", {}).get("validation")
                or split.get("split_roles", {}).get("test")
            )
            split["detected"] = has_train and has_holdout
        roles = {
            role: {
                "variables": list(split.get("split_roles", {}).get(role, [])),
                "uses": list(self.role_usage.get(role, [])),
            }
            for role in ("train", "validation", "test")
        }
        risks = []
        test_selection = [use for use in roles["test"]["uses"] if use.get("purpose") == "test selection"]
        if test_selection:
            selection_types = {use.get("selection_type") for use in test_selection}
            if selection_types == {"model_comparison"}:
                risk_type = "Test-set Used for Model Comparison"
                verification = "Confirmed by distinct estimator constructors in the evaluation loop"
                selection_type = "model_comparison"
            elif selection_types == {"hyperparameter_search"}:
                risk_type = "Test-set Used for Hyperparameter Comparison"
                verification = "Confirmed by loop variables passed to estimator parameters"
                selection_type = "hyperparameter_search"
            else:
                risk_type = "Repeated Test-set Evaluation Requires Review"
                verification = "Repeated use confirmed; comparison semantics require review"
                selection_type = "unknown"
            risks.append({
                "rule_id": "CODE_TEST_SET_OVERUSE",
                "type": risk_type,
                "severity": "High" if selection_type != "unknown" else "Medium",
                "verification": verification,
                "selection_type": selection_type,
                "evidence": test_selection,
            })
        repeated_test_evaluation = [
            use for use in roles["test"]["uses"]
            if use.get("purpose") == "metric"
        ]
        if len(repeated_test_evaluation) > 1 and not test_selection:
            risks.append({
                "rule_id": "CODE_REPEATED_TEST_EVALUATION_RISK",
                "type": "Repeated Test-set Evaluation Requires Review",
                "severity": "Low",
                "verification": "Manual confirmation required",
                "evidence": repeated_test_evaluation,
            })
        validation_selection = [
            use for use in roles["validation"]["uses"]
            if use.get("purpose") == "test selection"
        ]
        if validation_selection:
            risks.append({
                "rule_id": "CODE_VALIDATION_OVERFITTING_RISK",
                "type": "Validation Overfitting Risk",
                "severity": "Medium",
                "evidence": validation_selection,
            })
        flow["split_role_analysis"] = {
            "train_detected": bool(roles["train"]["variables"]),
            "validation_detected": bool(roles["validation"]["variables"]),
            "test_detected": bool(roles["test"]["variables"]),
            "split_roles": roles,
            "risks": risks,
        }

        fairness = flow["fairness"]
        fairness_events = [*fairness["dataset_metrics"], *fairness["prediction_metrics"]]
        fairness["evaluation_exists"] = bool(fairness_events)
        mitigation_sequences = [item["sequence"] for item in fairness["mitigations"]]
        fairness["before_after_comparison"] = bool(
            mitigation_sequences
            and any(event["sequence"] < min(mitigation_sequences) for event in fairness_events)
            and any(event["sequence"] > max(mitigation_sequences) for event in fairness_events)
        )
        unique_fairness_methods = {event["method"] for event in fairness_events}
        if not fairness_events:
            fairness["coverage"] = "No fairness evaluation detected"
        elif len(unique_fairness_methods) == 1:
            fairness["coverage"] = "Fairness evaluation exists, but metric coverage is limited."
        else:
            fairness["coverage"] = "Fairness evaluation detected"
        flow["evaluation_detected"] = bool(flow["metrics"] or flow["predict"] or fairness_events)
        self._infer_task_and_coverage()
        return flow

    def _infer_task_and_coverage(self):
        flow = self.flow
        evidence = []
        classification = False
        regression = False
        for model in flow["models"]:
            kind = model["type"]
            if kind.endswith("Classifier"):
                classification = True
                evidence.append(kind)
            if kind.endswith("Regressor") or kind in {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}:
                regression = True
                evidence.append(kind)
        metric_families = {item["family"] for item in flow["metrics"]}
        classification |= "Classification" in metric_families
        regression |= "Regression" in metric_families
        evidence.extend(item["name"] for item in flow["metrics"])
        if classification and not regression:
            task = "Classification"
            confidence = "High" if metric_families else "Medium"
        elif regression and not classification:
            task = "Regression"
            confidence = "High" if metric_families else "Medium"
        else:
            task = "Unknown"
            confidence = "Low"
        flow["task_type"] = {"type": task, "confidence": confidence, "evidence": evidence}

        labels = [item["label"] for item in flow["metrics"]]
        unique = list(dict.fromkeys(labels))
        if not unique:
            status = "No explicit metric detected" if flow["fit"] else "Unable to verify"
            suggested = []
        elif task == "Classification" and set(unique) == {"accuracy"}:
            status = "Metric Coverage Limited"
            suggested = ["precision", "recall", "f1"]
        elif task == "Regression" and len(unique) == 1:
            status = "Metric Coverage Limited"
            suggested = [item for item in ["rmse", "r2", "mae"] if item not in unique]
        else:
            status = "Evaluation detected"
            suggested = []
        flow["metric_coverage"] = {"status": status, "detected": unique, "suggested": suggested}


def analyze_python_data_flow(code, file_name=None):
    builder = _FlowBuilder(file_name)
    builder.process(code)
    return builder.finish()


def analyze_notebook_data_flow(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    code_cells = [
        (index, cell) for index, cell in enumerate(notebook.cells, 1)
        if cell.cell_type == "code"
    ]
    execution = [
        (index, cell.execution_count) for index, cell in code_cells
        if isinstance(cell.execution_count, int)
    ]
    counts = [count for _, count in execution]
    hidden_state = any(current <= previous for previous, current in zip(counts, counts[1:]))
    if len(execution) == len(code_cells) and len(set(counts)) == len(counts):
        ordered = sorted(code_cells, key=lambda item: item[1].execution_count)
    else:
        ordered = code_cells
    builder = _FlowBuilder(Path(notebook_path).name)
    for cell_index, cell in ordered:
        builder.process(preprocess_notebook_source(cell.source)["code"], cell=cell_index)
    flow = builder.finish()
    flow["notebook"] = {
        "execution_order": [index for index, _ in ordered],
        "hidden_state_risk": hidden_state,
        "execution_counts": counts,
    }
    return flow


def enrich_model_features(flow, data_result):
    """Resolve drop-based feature selections against actual dataset columns."""
    enriched = deepcopy(flow or _empty_flow())
    columns = list((data_result or {}).get("data_types", {}))
    excluded = set(enriched.get("excluded_features", []))
    target_column = (enriched.get("target") or {}).get("column")
    if not enriched.get("model_features") and columns and excluded:
        enriched["model_features"] = [
            column for column in columns if column not in excluded and column != target_column
        ]
    elif not enriched.get("model_features"):
        matrix = enriched.get("feature_matrix")
        selected = enriched.get("variable_columns", {}).get(matrix, [])
        if selected and selected[0] != "__all_except__":
            enriched["model_features"] = list(selected)
    target_column = (enriched.get("target") or {}).get("column")
    if target_column and enriched.get("task_type", {}).get("type") == "Unknown":
        dtype = str((data_result or {}).get("data_types", {}).get(target_column, ""))
        stats = (data_result or {}).get("numeric_statistics", {}).get(target_column, {})
        unique_count = stats.get("unique_count")
        if any(token in dtype for token in ("object", "string", "category", "bool")):
            task = "Classification"
        elif unique_count is not None and unique_count <= 20:
            task = "Classification"
        elif any(token in dtype for token in ("int", "float", "double")):
            task = "Regression"
        else:
            task = "Unknown"
        if task != "Unknown":
            enriched["task_type"] = {
                "type": task,
                "confidence": "Medium",
                "evidence": [f"target dtype={dtype}", f"unique_count={unique_count}"],
            }
    rows = int((data_result or {}).get("shape", {}).get("rows", 0) or 0)
    test_size = enriched.get("split", {}).get("test_size")
    if rows and isinstance(test_size, (int, float)) and not isinstance(test_size, bool):
        if 0 < float(test_size) < 1:
            estimated_test_rows = math.ceil(rows * float(test_size))
            test_ratio = float(test_size)
        else:
            estimated_test_rows = int(test_size)
            test_ratio = estimated_test_rows / rows
        role_analysis = enriched.setdefault("split_role_analysis", {})
        role_analysis["estimated_test_rows"] = estimated_test_rows
        role_analysis["test_ratio"] = test_ratio
        if estimated_test_rows < 20 or test_ratio < 0.05:
            risks = role_analysis.setdefault("risks", [])
            if not any(item.get("rule_id") == "CODE_INSUFFICIENT_TEST_SAMPLE" for item in risks):
                risks.append({
                    "rule_id": "CODE_INSUFFICIENT_TEST_SAMPLE",
                    "type": "Insufficient Final Test Sample",
                    "severity": "Medium" if estimated_test_rows < 20 else "Low",
                    "evidence": f"estimated_test_rows={estimated_test_rows}, test_ratio={test_ratio:.2%}",
                })
    if enriched.get("files"):
        enriched["files"] = [
            enrich_model_features({**candidate, "files": []}, data_result)
            for candidate in enriched["files"]
        ]
    return enriched


def merge_data_flows(flows):
    """Build a project view while retaining file-level replay boundaries."""
    candidates = list(flows or [])
    if not candidates:
        return {**_empty_flow(), "files": []}

    def score(flow):
        return (
            3 * bool(flow.get("split", {}).get("detected"))
            + 2 * bool(flow.get("fit"))
            + 2 * bool(flow.get("predict"))
            + 2 * bool(flow.get("metrics"))
            + bool(flow.get("model_features"))
        )

    primary = deepcopy(max(candidates, key=score))
    primary["files"] = deepcopy(candidates)
    for key in ("source_data", "preprocessing", "models", "fit", "predict", "metrics"):
        primary[key] = [deepcopy(item) for flow in candidates for item in flow.get(key, [])]
    primary["evaluation_detected"] = any(flow.get("evaluation_detected") for flow in candidates)

    roles = {role: {"variables": [], "uses": []} for role in ("train", "validation", "test")}
    risks = []
    for flow in candidates:
        analysis = flow.get("split_role_analysis", {})
        for role in roles:
            details = analysis.get("split_roles", {}).get(role, {})
            roles[role]["variables"].extend(details.get("variables", []))
            roles[role]["uses"].extend(details.get("uses", []))
        risks.extend(analysis.get("risks", []))
    for role in roles:
        roles[role]["variables"] = list(dict.fromkeys(roles[role]["variables"]))
    repeated_test_evaluation = [
        use for use in roles["test"]["uses"]
        if use.get("purpose") == "metric"
    ]
    if (
        len(repeated_test_evaluation) > 1
        and not any(risk.get("rule_id") == "CODE_TEST_SET_OVERUSE" for risk in risks)
    ):
        risks.append({
            "rule_id": "CODE_REPEATED_TEST_EVALUATION_RISK",
            "type": "Repeated Test-set Evaluation Requires Review",
            "severity": "Low",
            "verification": "Manual confirmation required",
            "evidence": repeated_test_evaluation,
        })
    primary["split_role_analysis"] = {
        "train_detected": bool(roles["train"]["variables"]),
        "validation_detected": bool(roles["validation"]["variables"]),
        "test_detected": bool(roles["test"]["variables"]),
        "split_roles": roles,
        "risks": risks,
    }
    primary["split"]["detected"] = any(
        flow.get("split", {}).get("detected") for flow in candidates
    )

    fairness = primary.get("fairness", {})
    for key in ("dataset_metrics", "prediction_metrics", "mitigations"):
        fairness[key] = [
            deepcopy(item) for flow in candidates
            for item in flow.get("fairness", {}).get(key, [])
        ]
    fairness["evaluation_exists"] = bool(
        fairness["dataset_metrics"] or fairness["prediction_metrics"]
    )
    fairness["before_after_comparison"] = any(
        flow.get("fairness", {}).get("before_after_comparison") for flow in candidates
    )
    primary["fairness"] = fairness

    labels = list(dict.fromkeys(
        item.get("label") for item in primary["metrics"] if item.get("label")
    ))
    families = {item.get("family") for item in primary["metrics"]}
    task = (
        "Classification" if families == {"Classification"} else
        "Regression" if families == {"Regression"} else
        primary.get("task_type", {}).get("type", "Unknown")
    )
    if not labels:
        coverage, suggested = (
            "No explicit metric detected" if primary["fit"] else "Unable to verify", []
        )
    elif task == "Classification" and set(labels) == {"accuracy"}:
        coverage, suggested = "Metric Coverage Limited", ["precision", "recall", "f1"]
    elif task == "Regression" and len(labels) == 1:
        coverage = "Metric Coverage Limited"
        suggested = [item for item in ("rmse", "r2", "mae") if item not in labels]
    else:
        coverage, suggested = "Evaluation detected", []
    primary["task_type"] = {
        "type": task,
        "confidence": "High" if families else primary.get("task_type", {}).get("confidence", "Low"),
        "evidence": list(dict.fromkeys(
            item for flow in candidates for item in flow.get("task_type", {}).get("evidence", [])
        )),
    }
    primary["metric_coverage"] = {
        "status": coverage, "detected": labels, "suggested": suggested,
    }
    return primary


def _flow_for_dataset(flow, csv_path):
    """Select the file-level lineage that most clearly reads ``csv_path``."""
    candidates = list((flow or {}).get("files") or [flow or {}])
    wanted = Path(csv_path).name.casefold()

    def score(candidate):
        paths = [source.get("path") for source in candidate.get("source_data", [])]
        matches = sum(
            Path(str(path)).name.casefold() == wanted
            for path in paths if isinstance(path, str)
        )
        return (
            20 * matches
            + 4 * bool(candidate.get("split", {}).get("detected"))
            + 2 * bool(candidate.get("operations"))
            + 2 * bool(candidate.get("model_features") or candidate.get("selected_features"))
            + bool(candidate.get("target"))
        )

    return max(candidates, key=score) if candidates else (flow or {})


def _replay_selector(spec):
    spec = spec or {}
    if not spec.get("reconstructable"):
        raise ValueError(f"selector is not reconstructable: {spec.get('value')}")
    if spec.get("kind") == "slice":
        return slice(spec.get("start"), spec.get("stop"), spec.get("step"))
    return spec.get("value")


def _step_precedes_split(step, split):
    split_line = (split or {}).get("line")
    split_cell = (split or {}).get("cell")
    if split_cell is None:
        return split_line is None or (step.get("line") or 0) <= split_line
    if step.get("cell") is None:
        return False
    return (
        step.get("cell") < split_cell
        or (
            step.get("cell") == split_cell
            and (step.get("line") or 0) <= (split_line or 0)
        )
    )


def _required_replay_orders(flow, replay_steps):
    """Find transformations on the backward lineage of the split inputs."""
    split = (flow or {}).get("split", {})
    required = {
        value for value in (split.get("input_features"), split.get("input_target"))
        if isinstance(value, str) and value
    }
    required_orders = set()
    relevant_steps = [
        step for step in replay_steps if _step_precedes_split(step, split)
    ]
    for step in reversed(relevant_steps):
        outputs = []
        if isinstance(step.get("output"), str):
            outputs.append(step["output"])
        outputs.extend(
            value for value in (step.get("outputs") or []) if isinstance(value, str)
        )
        matched = required.intersection(outputs)
        if not matched:
            continue
        if step.get("order") is not None:
            required_orders.add(step["order"])
        required.difference_update(outputs)
        inputs = []
        if isinstance(step.get("input"), str):
            inputs.append(step["input"])
        inputs.extend(
            value for value in (step.get("inputs") or []) if isinstance(value, str)
        )
        required.update(inputs)
    return required_orders


def _standard_scaler_fit_transform(source, options):
    """Replay sklearn's default StandardScaler math without running user code."""
    options = options or {}
    positional = list(options.get("positional") or [])
    keyword_values = dict(options.get("keywords") or {})
    keyword_safety = dict(options.get("keyword_reconstructable") or {})
    allowed = {"copy", "with_mean", "with_std"}
    if positional or set(keyword_values) - allowed:
        return None
    if any(not keyword_safety.get(name, False) for name in keyword_values):
        return None
    with_mean = keyword_values.get("with_mean", True)
    with_std = keyword_values.get("with_std", True)
    if not isinstance(with_mean, bool) or not isinstance(with_std, bool):
        return None
    try:
        numeric = source.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError):
        return None
    transformed = numeric.copy()
    if with_mean:
        transformed = transformed - numeric.mean(axis=0, skipna=True)
    if with_std:
        scale = numeric.std(axis=0, skipna=True, ddof=0).replace(0, 1).fillna(1)
        transformed = transformed / scale
    return transformed


def _dummy_column_lineage(
    source, transformed, source_lineage, columns=None, prefix=None, prefix_sep="_",
):
    """Map reconstructed dummy columns back to their original source fields."""
    source_columns = list(source.columns)
    encoded = list(columns) if columns is not None else [
        column for column in source_columns
        if str(source[column].dtype) in {"object", "string", "category", "bool", "boolean"}
    ]
    if prefix is None:
        prefixes = {column: str(column) for column in encoded}
    elif isinstance(prefix, str) and len(encoded) == 1:
        prefixes = {encoded[0]: prefix}
    elif isinstance(prefix, list) and len(prefix) == len(encoded):
        prefixes = dict(zip(encoded, map(str, prefix)))
    else:
        prefixes = {}
    output = {}
    for column in transformed.columns:
        if column in source_columns and column not in encoded:
            output[column] = list(source_lineage.get(column, [column]))
            continue
        matches = [
            source_column for source_column in encoded
            if source_column in prefixes
            and str(column).startswith(f"{prefixes[source_column]}{prefix_sep}")
        ]
        if matches:
            source_column = max(matches, key=lambda item: len(str(item)))
            output[column] = list(source_lineage.get(source_column, [source_column]))
        else:
            output[column] = []
    return output


def reconstruct_modeling_data(flow, csv_path, data_result=None):
    """Safely replay supported dataframe lineage and recover split membership.

    This executes no project code.  It only applies AST-recorded loads,
    aliases, column selections, column drops, row removals and literal slices
    to a private dataframe copy.
    """
    from data_analyzer import read_csv_for_analysis

    selected_flow = _flow_for_dataset(flow, csv_path)
    base = {
        "status": "Unable to reconstruct",
        "row_reconstruction": "Unable to verify",
        "reason": None,
        "source_csv": Path(csv_path).name,
        "source_variable": None,
        "raw_rows": None,
        "cleaned_rows": None,
        "model_input_variable": None,
        "model_input_columns": [],
        "model_input_source_columns": [],
        "model_input_feature_lineage": {},
        "field_reconstruction": "Exact",
        "unresolved_field_operations": [],
        "target_column": (selected_flow.get("target") or {}).get("column"),
        "train_indices": [],
        "test_indices": [],
        "train_rows": None,
        "test_rows": None,
        "evidence_level": "Unable to verify",
        "operation_impacts": [],
        "flow_file": selected_flow.get("file"),
    }
    try:
        original = read_csv_for_analysis(csv_path)
    except Exception as exc:
        return {**base, "reason": f"CSV read failed: {exc}"}
    base["raw_rows"] = len(original)
    frames = {}
    frame_lineage = {}
    replay_steps = list(selected_flow.get("replay_steps", []))
    required_replay_orders = _required_replay_orders(selected_flow, replay_steps)
    wanted = Path(csv_path).name.casefold()
    matched_load = False
    split = selected_flow.get("split", {})

    for step in replay_steps:
        operation = step.get("operation")
        input_name = step.get("input")
        output_name = step.get("output")
        step_required = step.get("order") in required_replay_orders
        if operation == "observe":
            if step_required:
                return {
                    **base,
                    "reason": (
                        f"observation result {step.get('output')} is used as modeling data "
                        "and cannot be safely replayed"
                    ),
                }
            continue
        if operation == "unsupported":
            if step_required:
                return {
                    **base,
                    "reason": (
                        f"cannot safely replay {step.get('output')}: {step.get('expression')} "
                        f"({step.get('reason')})"
                    ),
                }
            continue
        if operation == "load":
            source_path = step.get("path")
            if source_path is None or Path(str(source_path)).name.casefold() == wanted:
                frames[output_name] = original.copy()
                frame_lineage[output_name] = {
                    column: [column] for column in original.columns
                }
                matched_load = True
                base["source_variable"] = output_name
            continue
        if operation == "assign":
            if input_name in frames:
                frames[output_name] = frames[input_name].copy()
                frame_lineage[output_name] = deepcopy(frame_lineage.get(input_name, {}))
            continue
        if operation == "set_column":
            source = frames.get(input_name)
            if source is not None:
                updated = source.copy()
                if step.get("reconstructable"):
                    updated[step.get("column")] = step.get("value")
                else:
                    # Preserve row membership and column position without
                    # pretending the derived feature values are known.
                    updated[step.get("column")] = 0
                    base["field_reconstruction"] = "Partial"
                    base["unresolved_field_operations"].append({
                        "column": step.get("column"),
                        "expression": step.get("expression"),
                        "line": step.get("line"), "cell": step.get("cell"),
                    })
                frames[output_name] = updated
                lineage = deepcopy(frame_lineage.get(input_name, {}))
                lineage[step.get("column")] = []
                frame_lineage[output_name] = lineage
            continue
        if operation == "astype_column":
            source = frames.get(input_name)
            if source is None:
                continue
            if not step.get("reconstructable"):
                updated = source.copy()
                source_column = step.get("source_column")
                target_column = step.get("column")
                if source_column in source.columns:
                    updated[target_column] = source[source_column]
                frames[output_name] = updated
                lineage = deepcopy(frame_lineage.get(input_name, {}))
                lineage[target_column] = list(lineage.get(source_column, [source_column]))
                frame_lineage[output_name] = lineage
                base["field_reconstruction"] = "Partial"
                base["unresolved_field_operations"].append({
                    "operation": "astype", "column": target_column,
                    "expression": step.get("expression"),
                    "reason": "row lineage is preserved but the dtype is not statically known",
                    "line": step.get("line"), "cell": step.get("cell"),
                })
                continue
            source_column = step.get("source_column")
            target_column = step.get("column")
            if source_column not in source.columns:
                return {
                    **base,
                    "reason": f"type-conversion source column is unavailable: {source_column}",
                }
            try:
                updated = source.copy()
                updated[target_column] = updated[source_column].astype(step.get("dtype"))
                frames[output_name] = updated
                lineage = deepcopy(frame_lineage.get(input_name, {}))
                lineage[target_column] = list(lineage.get(source_column, [source_column]))
                frame_lineage[output_name] = lineage
            except Exception as exc:
                return {**base, "reason": f"type-conversion replay failed: {exc}"}
            continue
        if operation == "astype":
            source = frames.get(input_name)
            if source is None:
                continue
            input_columns = list(step.get("input_columns") or [])
            if input_columns:
                if not all(column in source.columns for column in input_columns):
                    return {
                        **base,
                        "reason": f"type-conversion columns are unavailable: {input_columns}",
                    }
                source = source[input_columns].copy()
            if not step.get("reconstructable"):
                frames[output_name] = source.copy()
                source_lineage = frame_lineage.get(input_name, {})
                frame_lineage[output_name] = {
                    column: list(source_lineage.get(column, [column]))
                    for column in source.columns
                }
                base["field_reconstruction"] = "Partial"
                base["unresolved_field_operations"].append({
                    "operation": "astype", "expression": step.get("expression"),
                    "reason": "row lineage is preserved but the dtype is not statically known",
                    "line": step.get("line"), "cell": step.get("cell"),
                })
                continue
            try:
                frames[output_name] = source.astype(step.get("dtype")).copy()
                source_lineage = frame_lineage.get(input_name, {})
                frame_lineage[output_name] = {
                    column: list(source_lineage.get(column, [column]))
                    for column in frames[output_name].columns
                }
            except Exception as exc:
                return {**base, "reason": f"type-conversion replay failed: {exc}"}
            continue
        if operation == "get_dummies":
            source = frames.get(input_name)
            if source is not None:
                input_columns = list(step.get("input_columns") or [])
                if input_columns:
                    if not all(column in source.columns for column in input_columns):
                        return {
                            **base,
                            "reason": f"get_dummies input columns are unavailable: {input_columns}",
                        }
                    source = source[input_columns].copy()
                if not step.get("reconstructable", True):
                    frames[output_name] = source.copy()
                    source_lineage = frame_lineage.get(input_name, {})
                    frame_lineage[output_name] = {
                        column: list(source_lineage.get(column, [column]))
                        for column in source.columns
                    }
                    base["field_reconstruction"] = "Partial"
                    base["unresolved_field_operations"].append({
                        "operation": "get_dummies",
                        "reason": (
                            "row lineage is preserved but one or more encoding "
                            "arguments are not statically known"
                        ),
                        "line": step.get("line"), "cell": step.get("cell"),
                    })
                    continue
                try:
                    dummy_options = {
                        "columns": step.get("columns"),
                        "drop_first": bool(step.get("drop_first")),
                        "dummy_na": bool(step.get("dummy_na")),
                    }
                    for option in ("prefix", "prefix_sep", "sparse", "dtype"):
                        if step.get(option) is not None:
                            dummy_options[option] = step[option]
                    frames[output_name] = pd.get_dummies(source, **dummy_options)
                    source_lineage = frame_lineage.get(input_name, {})
                    frame_lineage[output_name] = _dummy_column_lineage(
                        source, frames[output_name], source_lineage,
                        columns=step.get("columns"),
                        prefix=step.get("prefix"),
                        prefix_sep=(
                            "_" if step.get("prefix_sep") is None
                            else step.get("prefix_sep")
                        ),
                    )
                except Exception as exc:
                    return {**base, "reason": f"get_dummies replay failed: {exc}"}
            continue
        if operation == "preprocess_transform":
            source = frames.get(input_name)
            if source is None:
                continue
            if not step.get("row_preserving"):
                if step_required:
                    return {
                        **base,
                        "reason": (
                            f"cannot safely establish row lineage for "
                            f"{step.get('expression')}"
                        ),
                    }
                continue
            transformed = None
            if (
                step.get("transformer_type") == "StandardScaler"
                and step.get("method") == "fit_transform"
            ):
                transformed = _standard_scaler_fit_transform(
                    source, step.get("options"),
                )
            if transformed is None:
                # Known sklearn preprocessors preserve row membership for
                # fit_transform/transform.  A private copy is sufficient for
                # split reconstruction, but must never be presented as exact
                # transformed feature values.
                transformed = source.copy()
                base["field_reconstruction"] = "Partial"
                base["unresolved_field_operations"].append({
                    "operation": step.get("method"),
                    "transformer": step.get("transformer_type"),
                    "expression": step.get("expression"),
                    "reason": "row lineage is preserved but transformed feature values are not reconstructed",
                    "line": step.get("line"), "cell": step.get("cell"),
                })
            frames[output_name] = transformed
            source_lineage = frame_lineage.get(input_name, {})
            frame_lineage[output_name] = {
                column: list(source_lineage.get(column, [column]))
                for column in transformed.columns
            }
            continue
        if operation == "column_selection":
            source = frames.get(input_name)
            columns = list(step.get("columns") or [])
            if source is not None and all(column in source.columns for column in columns):
                frames[output_name] = source[columns].copy()
                source_lineage = frame_lineage.get(input_name, {})
                frame_lineage[output_name] = {
                    column: list(source_lineage.get(column, [column]))
                    for column in columns
                }
            continue
        if operation == "drop_columns":
            source = frames.get(input_name)
            columns = list(step.get("columns") or [])
            if source is not None and all(column in source.columns for column in columns):
                frames[output_name] = source.drop(columns=columns).copy()
                source_lineage = frame_lineage.get(input_name, {})
                frame_lineage[output_name] = {
                    column: list(source_lineage.get(column, [column]))
                    for column in frames[output_name].columns
                }
            continue
        if operation in {"dropna", "drop_duplicates"}:
            source = frames.get(input_name)
            if source is None:
                continue
            input_columns = list(step.get("input_columns") or [])
            if input_columns:
                if not all(column in source.columns for column in input_columns):
                    base["operation_impacts"].append({
                        "operation": operation,
                        "quantification": "Unable to quantify",
                        "reason": f"selected columns not present on {input_name}: {input_columns}",
                        "line": step.get("line"), "cell": step.get("cell"),
                    })
                    continue
                source = source[input_columns].copy()
            subset = list(step.get("subset") or [])
            missing = [column for column in subset if column not in source.columns]
            if missing:
                base["operation_impacts"].append({
                    "operation": operation,
                    "quantification": "Unable to quantify",
                    "reason": f"subset columns not present on {input_name}: {missing}",
                    "line": step.get("line"), "cell": step.get("cell"),
                })
                continue
            before = len(source)
            cleaned = (
                source.dropna(subset=subset or None)
                if operation == "dropna" else
                source.drop_duplicates(subset=subset or None)
            )
            affected = (
                [column for column in (subset or list(source.columns)) if source[column].isna().any()]
                if operation == "dropna" else list(subset or source.columns)
            )
            model_columns = set(selected_flow.get("model_features", []))
            target = (selected_flow.get("target") or {}).get("column")
            if target:
                model_columns.add(target)
            removed_index = source.index.difference(cleaned.index)
            base["operation_impacts"].append({
                "operation": operation,
                "quantification": "Exact",
                "evidence_level": "Verified from imported data + reconstructed operation",
                "calculation_scope": (
                    f"{input_name} columns={list(source.columns)}; operation={operation}; "
                    f"subset={subset or list(source.columns)}"
                ),
                "before": before,
                "after": len(cleaned),
                "removed": before - len(cleaned),
                "removal_rate": (before - len(cleaned)) / before if before else 0.0,
                "affected_columns": affected,
                "relevant_columns": [column for column in affected if column in model_columns],
                "non_model_columns": [column for column in affected if column not in model_columns],
                "sample_removed_rows": [
                    int(index) if isinstance(index, int) else str(index)
                    for index in removed_index[:10]
                ],
                "line": step.get("line"), "cell": step.get("cell"),
            })
            destination = output_name or (input_name if step.get("inplace") else None)
            if destination:
                frames[destination] = cleaned.copy()
                source_lineage = frame_lineage.get(input_name, {})
                frame_lineage[destination] = {
                    column: list(source_lineage.get(column, [column]))
                    for column in cleaned.columns
                }
            continue
        if operation == "row_slice":
            source = frames.get(input_name)
            spec = step.get("slice") or {}
            if source is not None and spec.get("reconstructable"):
                frames[output_name] = source.iloc[
                    slice(spec.get("start"), spec.get("stop"), spec.get("step"))
                ].copy()
                frame_lineage[output_name] = deepcopy(frame_lineage.get(input_name, {}))
            continue
        if operation in {"iloc_selection", "loc_selection"}:
            source = frames.get(input_name)
            if source is None:
                continue
            try:
                rows = _replay_selector(step.get("row_selector"))
                columns = _replay_selector(step.get("column_selector"))
                accessor = source.iloc if operation == "iloc_selection" else source.loc
                selected = accessor[rows, columns]
                frames[output_name] = selected.copy()
                selected_columns = (
                    list(selected.columns) if isinstance(selected, pd.DataFrame)
                    else [selected.name]
                )
                source_lineage = frame_lineage.get(input_name, {})
                frame_lineage[output_name] = {
                    column: list(source_lineage.get(column, [column]))
                    for column in selected_columns
                }
            except Exception as exc:
                return {
                    **base,
                    "reason": f"cannot replay {operation} for {output_name}: {exc}",
                }

    # Tests and direct API users may provide only the transformations, with
    # the CSV passed separately.  Bind one unambiguous external dataframe root
    # to that explicit input, then replay from the beginning.
    if not matched_load:
        required_steps = [
            step for step in replay_steps
            if step.get("order") in required_replay_orders
        ]
        if not required_steps:
            required_steps = list(replay_steps)
        referenced = [
            step.get("input") for step in required_steps
            if step.get("input") and step.get("input") not in frames
        ]
        produced = {
            value
            for step in required_steps
            for value in [step.get("output"), *(step.get("outputs") or [])]
            if isinstance(value, str)
        }
        roots = [
            name for name in dict.fromkeys(referenced)
            if isinstance(name, str) and name.isidentifier() and name not in produced
        ]
        root = roots[0] if len(roots) == 1 else ("df" if "df" in roots else None)
        if root:
            rebound = deepcopy(selected_flow)
            rebound["source_data"] = [{"variable": root, "path": str(csv_path)}]
            rebound["replay_steps"] = [
                {"order": 0, "operation": "load", "output": root, "path": str(csv_path)},
                *replay_steps,
            ]
            return reconstruct_modeling_data({"files": [rebound]}, csv_path, data_result)

    model_variable = split.get("input_features") or selected_flow.get("feature_matrix")
    model_frame = frames.get(model_variable)
    if isinstance(model_frame, pd.Series):
        model_frame = model_frame.to_frame()
    if model_frame is None:
        return {**base, "reason": f"model input variable {model_variable or '?'} could not be replayed"}
    base["model_input_variable"] = model_variable
    base["model_input_columns"] = list(model_frame.columns)
    model_lineage = frame_lineage.get(model_variable, {})
    base["model_input_feature_lineage"] = {
        str(column): list(model_lineage.get(column, []))
        for column in model_frame.columns
    }
    base["model_input_source_columns"] = list(dict.fromkeys(
        source_column
        for column in model_frame.columns
        for source_column in model_lineage.get(column, [])
    ))
    base["cleaned_rows"] = len(model_frame)

    method = split.get("method") or ("train_test_split" if split.get("calls") else None)
    try:
        if method == "train_test_split":
            if split.get("random_state") is None:
                return {**base, "reason": "train_test_split random_state is unknown"}
            if split.get("train_size_provided"):
                return {**base, "reason": "train_test_split train_size replay is not supported"}
            if (
                split.get("test_size_provided")
                and not split.get("test_size_reconstructable", True)
            ):
                return {**base, "reason": "train_test_split test_size is unknown"}
            from sklearn.model_selection import train_test_split

            indices = list(model_frame.index)
            stratify = None
            if split.get("stratify"):
                target_variable = split.get("input_target")
                target_values = frames.get(target_variable)
                if isinstance(target_values, pd.DataFrame) and target_values.shape[1] == 1:
                    target_values = target_values.iloc[:, 0]
                if isinstance(target_values, pd.Series):
                    stratify = target_values.loc[indices]
                elif base.get("target_column") in original.columns:
                    stratify = original.loc[indices, base["target_column"]]
                else:
                    return {
                        **base,
                        "reason": f"stratify expression {split.get('stratify')} could not be replayed",
                    }
            train_index, test_index = train_test_split(
                indices,
                test_size=split.get("test_size") if split.get("test_size") is not None else 0.25,
                random_state=split.get("random_state"),
                stratify=stratify,
            )
        elif method == "slice":
            train_names = split.get("split_roles", {}).get("train", [])
            test_names = split.get("split_roles", {}).get("test", [])
            train_objects = [frames[name] for name in train_names if name in frames]
            test_objects = [frames[name] for name in test_names if name in frames]
            if not train_objects or not test_objects:
                return {
                    **base,
                    "reason": "train/test slice bounds or source variables could not be replayed",
                }
            train_index = list(train_objects[0].index)
            test_index = list(test_objects[0].index)
        else:
            return {
                **base,
                "reason": f"split method {method or '?'} is not safely reconstructable",
            }
    except Exception as exc:
        return {**base, "reason": f"split replay failed: {exc}"}
    values_exact = base.get("field_reconstruction") == "Exact"
    return {
        **base,
        "status": "Exact" if values_exact else "Partial",
        "row_reconstruction": "Exact",
        "evidence_level": (
            "Verified from imported data + reconstructed operation"
            if values_exact else
            "Verified row lineage; transformed feature values are partially reconstructed"
        ),
        "reason": None,
        "train_indices": [int(value) if isinstance(value, int) else value for value in train_index],
        "test_indices": [int(value) if isinstance(value, int) else value for value in test_index],
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "model_input_frame": model_frame,
    }


def quantify_data_impacts(flow, csv_path, data_result=None):
    """Safely replay supported row-removal operations on an analysis-only DataFrame copy."""
    from data_analyzer import read_csv_for_analysis

    operations = list((flow or {}).get("operations", []))
    if not operations:
        return []
    reconstruction = reconstruct_modeling_data(flow, csv_path, data_result)
    replayed = list(reconstruction.get("operation_impacts", []))
    if replayed:
        return replayed
    return [{
        "operation": operation.get("type"),
        "quantification": "Unable to quantify",
        "reason": reconstruction.get("reason") or "operation lineage could not be replayed",
        "before": None,
        "after": None,
        "removed": None,
        "removal_rate": None,
        "relevant_columns": [],
        "line": operation.get("line"),
        "cell": operation.get("cell"),
    } for operation in operations]

    # Legacy fallback retained below for third-party callers pinned to older
    # flow dictionaries; canonical flows always return above.
    try:
        original = read_csv_for_analysis(csv_path)
    except Exception as exc:
        return [{
            "operation": "row filtering",
            "quantification": "Unable to quantify",
            "reason": str(exc),
            "before": None,
            "after": None,
            "removed": None,
            "removal_rate": None,
            "relevant_columns": [],
        }]

    model_features = list((flow or {}).get("model_features", []))
    target_column = ((flow or {}).get("target") or {}).get("column")
    model_columns = [column for column in [*model_features, target_column] if column in original.columns]
    variable_columns = (flow or {}).get("variable_columns", {})
    frames = {}
    for source in (flow or {}).get("source_data", []):
        if source.get("variable"):
            frames[source["variable"]] = original.copy()
    if not frames:
        frames["df"] = original.copy()
    for variable, columns in variable_columns.items():
        if not columns:
            continue
        if columns[0] == "__all_except__":
            selected = [column for column in original.columns if column not in set(columns[1:])]
        else:
            selected = [column for column in columns if column in original.columns]
        if selected:
            frames[variable] = original[selected].copy()

    impacts = []
    for operation in operations:
        receiver = operation.get("receiver")
        frame = frames.get(receiver)
        if frame is None:
            impacts.append({
                "operation": operation.get("type"),
                "quantification": "Unable to quantify",
                "reason": f"无法把变量 {receiver or '?'} 映射到所选 CSV。",
                "before": None,
                "after": None,
                "removed": None,
                "removal_rate": None,
                "relevant_columns": [],
                "line": operation.get("line"),
                "cell": operation.get("cell"),
            })
            continue
        before = len(frame)
        operation_type = operation.get("type")
        subset = [column for column in (operation.get("subset") or []) if column in frame.columns]
        try:
            if operation_type == "dropna":
                scope = subset or list(frame.columns)
                cleaned = frame.dropna(subset=scope)
                affected_columns = [column for column in scope if frame[column].isna().any()]
            elif operation_type == "drop_duplicates":
                scope = subset or list(frame.columns)
                cleaned = frame.drop_duplicates(subset=subset or None)
                affected_columns = scope
            else:
                raise ValueError(f"不支持的安全量化操作：{operation_type}")
        except Exception as exc:
            impacts.append({
                "operation": operation_type,
                "quantification": "Unable to quantify",
                "reason": str(exc),
                "before": before,
                "after": None,
                "removed": None,
                "removal_rate": None,
                "relevant_columns": [],
                "line": operation.get("line"),
                "cell": operation.get("cell"),
            })
            continue

        removed_indices = [int(index) if isinstance(index, int) else str(index)
                           for index in frame.index.difference(cleaned.index)[:20]]
        after = len(cleaned)
        removed = before - after
        relevant = [column for column in affected_columns if column in model_columns]
        excluded = [column for column in affected_columns if column not in model_columns]
        impact = {
            "operation": operation_type,
            "quantification": "Exact",
            "before": before,
            "after": after,
            "removed": removed,
            "removal_rate": removed / before if before else 0.0,
            "affected_columns": affected_columns,
            "relevant_columns": relevant,
            "non_model_columns": excluded,
            "sample_removed_rows": removed_indices[:10],
            "line": operation.get("line"),
            "cell": operation.get("cell"),
        }
        impacts.append(impact)
        output = operation.get("assigned_to") or (receiver if operation.get("inplace") else None)
        if output:
            frames[output] = cleaned.copy()
        if receiver and operation.get("inplace"):
            frames[receiver] = cleaned.copy()
    return impacts


def reconcile_code_issues_with_flow(code_result, data_flow):
    """Remove absence claims contradicted by project-level data-flow evidence."""
    flow = data_flow or {}
    split_detected = bool(flow.get("split", {}).get("detected"))
    evaluation_detected = bool(flow.get("evaluation_detected"))
    metrics_detected = bool(flow.get("metrics"))
    metric_labels = {item.get("label") for item in flow.get("metrics", []) if item.get("label")}
    holdout_evaluation = any(
        any(marker in str(event.get(key, "")).casefold() for marker in ("test", "val", "valid"))
        for event in [*flow.get("predict", []), *flow.get("metrics", [])]
        for key in ("X", "target", "prediction")
    )
    reconciled = []
    for location in code_result or []:
        issues = []
        for issue in location.get("issues", []):
            rule_id = issue.get("rule_id")
            if split_detected and rule_id == "CODE_NO_VALIDATION_SPLIT":
                continue
            if (evaluation_detected or metrics_detected) and rule_id == "CODE_NO_EVALUATION":
                continue
            if holdout_evaluation and rule_id == "CODE_TRAINING_ONLY_EVALUATION":
                continue
            if len(metric_labels) > 1 and rule_id in {
                "CODE_SINGLE_CLASSIFICATION_METRIC", "CODE_SINGLE_REGRESSION_METRIC",
            }:
                continue
            issues.append(issue)
        if issues:
            reconciled.append({**location, "issues": issues})
    return reconciled


def flow_summary_text(flow):
    """Render a compact factual flow without inventing missing stages."""
    flow = flow or {}
    parts = []
    sources = flow.get("source_data", [])
    if sources:
        parts.append(sources[0].get("variable") or "data")
    if flow.get("feature_matrix") or flow.get("target"):
        x_name = flow.get("feature_matrix") or "X"
        y_name = (flow.get("target") or {}).get("variable") or "y"
        parts.append(f"{x_name}/{y_name}")
    if flow.get("split", {}).get("detected"):
        parts.append("train_test_split")
    if flow.get("fit"):
        last_fit = flow["fit"][-1]
        parts.append(f"{last_fit.get('object') or 'model'}.fit({last_fit.get('X') or '?'})")
    if flow.get("predict"):
        last_predict = flow["predict"][-1]
        parts.append(f"predict({last_predict.get('X') or '?'})")
    if flow.get("metrics"):
        parts.append(", ".join(item["label"] for item in flow["metrics"]))
    return " → ".join(parts) if parts else "Unable to verify automatically."
