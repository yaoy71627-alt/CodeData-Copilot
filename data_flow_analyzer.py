"""Lightweight, evidence-based data flow analysis for data science code."""

from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path

import nbformat

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


def _role_from_name(name):
    value = str(name or "").casefold()
    if any(marker in value for marker in ("validation", "valid", "_val", "val_")):
        return "validation"
    if "test" in value:
        return "test"
    if "train" in value:
        return "train"
    return None


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
        "operations": [],
        "variable_columns": {},
        "lineage": [],
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
        self.role_usage = {"train": [], "validation": [], "test": []}
        self.sequence = 0

    def _record_assignment(self, node, cell):
        names = _assignment_names(node)
        if not names:
            return
        value = node.value
        literal = _literal_value(value, self.literals)
        if len(names) == 1 and literal is not None:
            self.literals[names[0]] = literal

        if isinstance(value, ast.Call):
            call_name = _call_name(value)
            short_name = call_name.split(".")[-1]
            if len(names) == 1:
                self.constructors[names[0]] = short_name
                self.object_types[names[0]] = short_name
                self.constructor_inputs[names[0]] = {
                    "positional": [_expr_label(arg) for arg in value.args],
                    "keywords": {item.arg: _expr_label(item.value) for item in value.keywords if item.arg},
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

            if short_name == "train_test_split" and len(names) >= 2:
                args = [_expr_label(arg) for arg in value.args]
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
                    "test_size": _constant_keyword(value, "test_size", self.literals),
                    "random_state": _constant_keyword(value, "random_state", self.literals),
                }
                split["calls"].append(split_event)
                split.update({
                    "detected": True,
                    "line": node.lineno,
                    "cell": cell,
                    "input_features": args[0] if args else None,
                    "input_target": args[1] if len(args) > 1 else None,
                    "train": train,
                    "test": test,
                    "test_size": _constant_keyword(value, "test_size", self.literals),
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

            if short_name in {"dropna", "drop_duplicates"} and len(names) == 1:
                receiver = _receiver_name(value)
                subset = _constant_keyword(value, "subset", self.literals)
                if isinstance(subset, str):
                    subset = [subset]
                self.flow["operations"].append({
                    "type": short_name,
                    "receiver": receiver,
                    "assigned_to": names[0],
                    "subset": subset,
                    "inplace": bool(_constant_keyword(value, "inplace", self.literals)),
                    **_event_location(node, self.flow["file"], cell),
                })
                if receiver in self.variable_columns:
                    self.variable_columns[names[0]] = list(self.variable_columns[receiver])

            if short_name == "predict" and len(names) == 1:
                prediction_input = _expr_label(value.args[0]) if value.args else None
                event = {
                    "object": _receiver_name(value), "X": prediction_input, "output": names[0],
                    **_event_location(node, self.flow["file"], cell),
                }
                self.flow["predict"].append(event)
                self.flow["lineage"].append({"from": prediction_input, "to": names[0], "operation": "predict"})

        if isinstance(value, ast.Name) and len(names) == 1:
            if value.id in self.variable_columns:
                self.variable_columns[names[0]] = list(self.variable_columns[value.id])
            self.flow["lineage"].append({"from": value.id, "to": names[0], "operation": "assign"})
            if value.id in self.object_types:
                self.object_types[names[0]] = self.object_types[value.id]

        if isinstance(value, ast.Subscript) and len(names) == 1:
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
        if short_name == "fit":
            purpose = "fit"
        elif short_name in {"fit_transform", "fit_resample"}:
            purpose = "transform fit"
        elif short_name == "predict":
            purpose = "predict"
        elif short_name in CLASSIFICATION_METRICS or short_name in REGRESSION_METRICS:
            purpose = "hyperparameter comparison" if in_loop else "metric"
        if purpose:
            for variable in argument_names:
                role = _role_from_name(variable)
                if role:
                    self.role_usage[role].append({
                        "variable": variable,
                        "purpose": purpose,
                        "object": receiver,
                        "in_loop": in_loop,
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

    def process(self, code, cell=None):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        assignments = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        for node in sorted(assignments, key=lambda item: (item.lineno, item.col_offset)):
            self._record_assignment(node, cell)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for call in sorted(calls, key=lambda item: (item.lineno, item.col_offset)):
            self._record_call(call, cell, parents)

    def finish(self):
        flow = self.flow
        flow["variable_columns"] = deepcopy(self.variable_columns)
        flow["object_types"] = deepcopy(self.object_types)
        flow["constructor_inputs"] = deepcopy(self.constructor_inputs)
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
        roles = {
            role: {
                "variables": list(split.get("split_roles", {}).get(role, [])),
                "uses": list(self.role_usage.get(role, [])),
            }
            for role in ("train", "validation", "test")
        }
        risks = []
        test_selection = [use for use in roles["test"]["uses"] if use.get("purpose") == "hyperparameter comparison"]
        if test_selection:
            risks.append({
                "rule_id": "CODE_TEST_SET_OVERUSE",
                "type": "Test-set Overuse / Model Selection Leakage",
                "severity": "High",
                "evidence": test_selection,
            })
        validation_selection = [
            use for use in roles["validation"]["uses"]
            if use.get("purpose") == "hyperparameter comparison"
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
    return enriched


def merge_data_flows(flows):
    """Choose a primary complete flow while retaining every file-level flow."""
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
    primary["files"] = candidates
    return primary


def quantify_data_impacts(flow, csv_path, data_result=None):
    """Safely replay supported row-removal operations on an analysis-only DataFrame copy."""
    from data_analyzer import read_csv_for_analysis

    operations = list((flow or {}).get("operations", []))
    if not operations:
        return []
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
    reconciled = []
    for location in code_result or []:
        issues = []
        for issue in location.get("issues", []):
            rule_id = issue.get("rule_id")
            if split_detected and rule_id == "CODE_NO_VALIDATION_SPLIT":
                continue
            if (evaluation_detected or metrics_detected) and rule_id == "CODE_NO_EVALUATION":
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
