"""P0 regressions for safe Titanic modeling-data reconstruction."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_flow_analyzer import (
    analyze_python_data_flow,
    quantify_data_impacts,
    reconstruct_modeling_data,
)


def _basic_flow(extra_lines="", *, random_state="13", test_size="0.25"):
    return analyze_python_data_flow(
        f"""
features = ["feature"]
model_data = raw[features + ["target"]].copy()
model_data = model_data.dropna().copy()
{extra_lines}
X = model_data[features]
y = model_data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)
"""
    )


def test_len_and_print_observations_do_not_interrupt_reconstruction(tmp_path):
    frame = pd.DataFrame({
        "feature": [1, 2, None, 4, 5, 6, 7, 8],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    csv_path = tmp_path / "observations.csv"
    frame.to_csv(csv_path, index=False)
    flow = _basic_flow("before = len(model_data)\nprint(before)")

    reconstruction = reconstruct_modeling_data(flow, csv_path)

    assert any(
        step["operation"] == "observe" and step["expression"] == "len(model_data)"
        for step in flow["replay_steps"]
    )
    assert reconstruction["status"] == "Exact"
    assert reconstruction["raw_rows"] == 8
    assert reconstruction["cleaned_rows"] == 7


def test_dropna_preserves_original_sample_indices_and_impact(tmp_path):
    frame = pd.DataFrame({
        "feature": [10, None, 12, 13, None, 15, 16, 17],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    csv_path = tmp_path / "dropna.csv"
    frame.to_csv(csv_path, index=False)
    flow = _basic_flow()

    reconstruction = reconstruct_modeling_data(flow, csv_path)
    impacts = quantify_data_impacts(flow, csv_path)

    surviving = {0, 2, 3, 5, 6, 7}
    assert set(reconstruction["train_indices"]) | set(
        reconstruction["test_indices"]
    ) == surviving
    assert set(reconstruction["train_indices"]).isdisjoint(
        reconstruction["test_indices"]
    )
    assert (impacts[0]["before"], impacts[0]["after"], impacts[0]["removed"]) == (
        8, 6, 2,
    )
    assert impacts[0]["sample_removed_rows"] == [1, 4]


def test_selection_astype_and_get_dummies_keep_row_lineage(tmp_path):
    frame = pd.DataFrame({
        "value": [1, 2, None, 4, 5, 6],
        "category": ["a", "b", "a", "c", "b", "c"],
        "target": [0, 1, 0, 1, 0, 1],
        "unused": range(6),
    })
    csv_path = tmp_path / "encoded.csv"
    frame.to_csv(csv_path, index=False)
    code = """
features = ["value", "category"]
model_data = raw[features + ["target"]].copy()
model_data = model_data.dropna()
model_data["category"] = model_data["category"].astype(str)
X = pd.get_dummies(
    model_data[features], columns=["category"], drop_first=True
)
y = model_data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=7
)
"""
    flow = analyze_python_data_flow(code)

    reconstruction = reconstruct_modeling_data(flow, csv_path)

    assert reconstruction["status"] == "Exact"
    assert reconstruction["field_reconstruction"] == "Exact"
    assert reconstruction["model_input_columns"] == [
        "value", "category_b", "category_c",
    ]
    assert reconstruction["model_input_feature_lineage"] == {
        "value": ["value"],
        "category_b": ["category"],
        "category_c": ["category"],
    }
    assert set(reconstruction["train_indices"]) | set(
        reconstruction["test_indices"]
    ) == {0, 1, 3, 4, 5}
    assert flow["model_features"] == ["value", "category"]


def test_standard_scaler_fit_transform_reconstructs_values_and_indices(tmp_path):
    frame = pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 8.0, 16.0],
        "b": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
        "target": [0, 1, 0, 1, 0, 1],
    })
    csv_path = tmp_path / "scaled.csv"
    frame.to_csv(csv_path, index=False)
    code = """
features = ["a", "b"]
model_data = raw[features + ["target"]].dropna().copy()
X = model_data[features]
y = model_data["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=5
)
"""
    reconstruction = reconstruct_modeling_data(
        analyze_python_data_flow(code), csv_path,
    )
    expected_source = frame.dropna()[["a", "b"]]
    expected = StandardScaler().fit_transform(expected_source)

    assert reconstruction["status"] == "Exact"
    assert reconstruction["row_reconstruction"] == "Exact"
    assert reconstruction["field_reconstruction"] == "Exact"
    assert reconstruction["model_input_columns"] == ["a", "b"]
    assert np.allclose(reconstruction["model_input_frame"].to_numpy(), expected)
    assert set(reconstruction["train_indices"] + reconstruction["test_indices"]) == {
        0, 1, 3, 4, 5,
    }


def test_row_lineage_is_exact_when_known_preprocessor_values_are_partial(tmp_path):
    frame = pd.DataFrame({
        "a": range(10), "b": range(10, 20), "target": [0, 1] * 5,
    })
    csv_path = tmp_path / "pca.csv"
    frame.to_csv(csv_path, index=False)
    code = """
X = df[["a", "b"]]
y = df["target"]
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.3, random_state=9
)
"""
    flow = analyze_python_data_flow(code)
    reconstruction = reconstruct_modeling_data(flow, csv_path)

    assert reconstruction["status"] == "Partial"
    assert reconstruction["row_reconstruction"] == "Exact"
    assert reconstruction["field_reconstruction"] == "Partial"
    assert (reconstruction["train_rows"], reconstruction["test_rows"]) == (7, 3)
    assert reconstruction["unresolved_field_operations"][0]["transformer"] == "PCA"


def test_unknown_encoding_arguments_keep_rows_but_not_exact_feature_scope(tmp_path):
    frame = pd.DataFrame({
        "category": ["a", "b", "a", "c", "b", "c"],
        "value": range(6),
        "target": [0, 1, 0, 1, 0, 1],
    })
    csv_path = tmp_path / "dynamic_encoding.csv"
    frame.to_csv(csv_path, index=False)
    code = """
columns_to_encode = choose_columns()
X = pd.get_dummies(df[["category", "value"]], columns=columns_to_encode)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=4
)
"""
    flow = analyze_python_data_flow(code)
    reconstruction = reconstruct_modeling_data(flow, csv_path)

    assert reconstruction["status"] == "Partial"
    assert reconstruction["row_reconstruction"] == "Exact"
    assert reconstruction["field_reconstruction"] == "Partial"
    assert (reconstruction["train_rows"], reconstruction["test_rows"]) == (3, 3)

    from data_analyzer import analyze_csv

    result = analyze_csv(csv_path, data_flow=flow)
    feature_scope = result["duplicate_split_analysis"]["scopes"]["model_input_row"]
    assert feature_scope["status"] == "Unable to quantify"
    assert feature_scope["cross_split_duplicate_groups"] is None


def test_fixed_seed_train_test_split_matches_sklearn_indices(tmp_path):
    frame = pd.DataFrame({
        "feature": range(20),
        "target": [0, 1] * 10,
    })
    csv_path = tmp_path / "split.csv"
    frame.to_csv(csv_path, index=False)
    code = """
features = ["feature"]
model_data = raw[features + ["target"]].copy()
X = model_data[features]
y = model_data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23, stratify=y
)
"""
    flow = analyze_python_data_flow(code)

    reconstruction = reconstruct_modeling_data(flow, csv_path)
    expected_train, expected_test = train_test_split(
        list(frame.index),
        test_size=0.25,
        random_state=23,
        stratify=frame["target"],
    )

    assert reconstruction["status"] == "Exact"
    assert reconstruction["train_indices"] == expected_train
    assert reconstruction["test_indices"] == expected_test


def test_unsafe_required_transformation_is_not_executed(tmp_path):
    frame = pd.DataFrame({"feature": range(8), "target": [0, 1] * 4})
    csv_path = tmp_path / "unsafe.csv"
    marker = tmp_path / "must_not_exist.txt"
    frame.to_csv(csv_path, index=False)
    code = f"""
model_data = raw[["feature", "target"]].copy()
X = model_data.apply(
    lambda row: open({str(marker)!r}, "w").write("executed"), axis=1
)
y = model_data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3
)
"""
    flow = analyze_python_data_flow(code)

    reconstruction = reconstruct_modeling_data(flow, csv_path)

    assert reconstruction["status"] == "Unable to reconstruct"
    assert "cannot safely replay X" in reconstruction["reason"]
    assert not marker.exists()


def test_unknown_random_seed_does_not_produce_exact_split(tmp_path):
    frame = pd.DataFrame({"feature": range(8), "target": [0, 1] * 4})
    csv_path = tmp_path / "unknown_seed.csv"
    frame.to_csv(csv_path, index=False)
    flow = _basic_flow(random_state="runtime_seed")

    reconstruction = reconstruct_modeling_data(flow, csv_path)

    assert reconstruction["status"] == "Unable to reconstruct"
    assert reconstruction["train_indices"] == []
    assert reconstruction["test_indices"] == []
    assert "random_state is unknown" in reconstruction["reason"]


def test_synthetic_titanic_equivalent_restores_expected_counts(tmp_path):
    row_count = 916
    kept_rows = 647
    frame = pd.DataFrame({
        "PassengerId": list(range(1, row_count + 1)),
        "Survived": [index % 2 for index in range(row_count)],
        "Pclass": [(index % 3) + 1 for index in range(row_count)],
        "Sex": ["female" if index % 2 else "male" for index in range(row_count)],
        "Age": [18 + (index % 60) for index in range(row_count)],
        "SibSp": [index % 4 for index in range(row_count)],
        "Parch": [(index // 4) % 3 for index in range(row_count)],
        "Fare": [20 + index / 100 for index in range(row_count)],
        "Embarked": [["S", "C", "Q"][index % 3] for index in range(row_count)],
    })
    frame.loc[kept_rows:, "Age"] = None
    expected_train, expected_test = train_test_split(
        list(range(kept_rows)),
        test_size=0.25,
        random_state=42,
        stratify=frame.loc[: kept_rows - 1, "Survived"],
    )
    for train_row, test_row in zip(expected_train[:9], expected_test[:9]):
        frame.at[test_row, "PassengerId"] = frame.at[train_row, "PassengerId"]

    csv_path = tmp_path / "titanic_equivalent.csv"
    frame.to_csv(csv_path, index=False)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    code = f"""
features = {features!r}
model_data = df[features + ["Survived"]].copy()
model_data = model_data.dropna().copy()
before = len(model_data)
X_encoded = pd.get_dummies(model_data[features], drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(X_encoded)
y = model_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
"""
    flow = analyze_python_data_flow(code, "titanic.ipynb")

    reconstruction = reconstruct_modeling_data(flow, csv_path)
    impacts = quantify_data_impacts(flow, csv_path)

    assert reconstruction["status"] == "Exact"
    assert (reconstruction["raw_rows"], reconstruction["cleaned_rows"]) == (916, 647)
    assert (impacts[0]["before"], impacts[0]["after"], impacts[0]["removed"]) == (
        916, 647, 269,
    )
    assert (reconstruction["train_rows"], reconstruction["test_rows"]) == (485, 162)
    assert reconstruction["train_indices"] == expected_train
    assert reconstruction["test_indices"] == expected_test
    assert reconstruction["field_reconstruction"] == "Exact"
    assert set(reconstruction["model_input_source_columns"]) == set(features)
    assert reconstruction["model_input_feature_lineage"]["Sex_male"] == ["Sex"]

    from data_analyzer import analyze_csv

    inferred_result = analyze_csv(csv_path, data_flow=flow)
    inferred = inferred_result["duplicate_split_analysis"]
    assert inferred["scopes"]["candidate_id"]["fields"] == ["PassengerId"]
    assert inferred["scopes"]["candidate_id"]["cross_split_duplicate_groups"] == 9
    inferred_rules = {issue["rule_id"] for issue in inferred_result["issues"]}
    assert "DQ_CROSS_SPLIT_CANDIDATE_ID_OVERLAP" in inferred_rules
    assert "DQ_CROSS_SPLIT_ENTITY_ID_OVERLAP" not in inferred_rules

    result = analyze_csv(
        csv_path,
        constraints={"business_id_columns": ["PassengerId"]},
        data_flow=flow,
    )
    assert result["duplicate_split_analysis"]["cross_split_duplicate_groups"] == 9
    assert "DQ_CROSS_SPLIT_ENTITY_ID_OVERLAP" in {
        issue["rule_id"] for issue in result["issues"]
    }


def test_real_titanic_expected_counts_when_case_data_is_available():
    project_root = Path(__file__).resolve().parents[1]
    required_columns = {
        "PassengerId", "Survived", "Pclass", "Sex", "Age",
        "SibSp", "Parch", "Fare", "Embarked",
    }
    csv_path = None
    configured = os.environ.get("CODEDATA_TITANIC_CSV")
    candidates = [Path(configured)] if configured else project_root.rglob("*.csv")
    for candidate in candidates:
        if not configured and "titanic" not in candidate.name.casefold():
            continue
        try:
            candidate_frame = pd.read_csv(candidate)
        except Exception:
            continue
        if len(candidate_frame) == 916 and required_columns.issubset(candidate_frame.columns):
            csv_path = candidate
            break
    if csv_path is None:
        if configured:
            pytest.fail("CODEDATA_TITANIC_CSV is unreadable or does not match the original 916-row case schema")
        pytest.skip("The 916-row Titanic case CSV is not available in this project.")

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    code = f"""
features = {features!r}
model_data = df[features + ["Survived"]].copy()
model_data = model_data.dropna().copy()
before = len(model_data)
X_encoded = pd.get_dummies(model_data[features], drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(X_encoded)
y = model_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
"""
    flow = analyze_python_data_flow(code, "titanic.ipynb")
    reconstruction = reconstruct_modeling_data(flow, csv_path)
    impacts = quantify_data_impacts(flow, csv_path)

    assert reconstruction["status"] == "Exact"
    assert reconstruction["raw_rows"] == 916
    assert reconstruction["cleaned_rows"] == 647
    assert (impacts[0]["before"], impacts[0]["after"], impacts[0]["removed"]) == (
        916, 647, 269,
    )
    assert (reconstruction["train_rows"], reconstruction["test_rows"]) == (485, 162)

    from data_analyzer import analyze_csv

    result = analyze_csv(
        csv_path,
        constraints={"business_id_columns": ["PassengerId"]},
        data_flow=flow,
    )
    assert result["duplicate_split_analysis"]["cross_split_duplicate_groups"] == 9
