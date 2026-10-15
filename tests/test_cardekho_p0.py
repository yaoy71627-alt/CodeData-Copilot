"""P0 regressions for duplicate analysis semantics used by the CarDekho case."""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

import agent
from data_analyzer import analyze_csv
from data_flow_analyzer import analyze_python_data_flow


def _split_flow(features, *, random_state=17):
    random_state_argument = (
        f", random_state={random_state}" if random_state is not None else ""
    )
    return analyze_python_data_flow(
        f"""
X = df[{features!r}]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5{random_state_argument}
)
"""
    )


def test_analysis_is_json_serializable_and_accepted_by_knowledge_retrieval(tmp_path):
    frame = pd.DataFrame({
        "vehicle_id": ["v1", "v1", "v2", "v2", "v3", "v3"],
        "mileage": [10, 11, 20, 21, 30, 31],
        "target": [100, 101, 200, 201, 300, 301],
    })
    csv_path = tmp_path / "vehicles.csv"
    frame.to_csv(csv_path, index=False)

    result = analyze_csv(
        csv_path,
        constraints={"business_id_columns": ["vehicle_id"]},
        data_flow=_split_flow(["mileage"]),
    )

    serialized = json.dumps(result, ensure_ascii=False)
    knowledge = agent.retrieve_relevant_knowledge(
        [], [{"file": csv_path.name, "result": result}]
    )

    assert serialized
    assert isinstance(knowledge, str)
    assert result["duplicate_split_analysis"]["scope"] == "business_id"


def test_repeated_model_names_are_not_deterministic_entity_leakage(tmp_path):
    frame = pd.DataFrame({
        "row_number": range(12),
        "car_name": ["sedan", "sedan", "sedan", "suv", "suv", "suv"] * 2,
        "listing_code": ["a", "a", "b", "b", "c", "c"] * 2,
        "target": range(12),
    })
    csv_path = tmp_path / "cars.csv"
    frame.to_csv(csv_path, index=False)

    result = analyze_csv(csv_path, data_flow=_split_flow(["car_name"]))
    duplicate = result["duplicate_split_analysis"]

    assert result["duplicate_rows"] == 0
    assert duplicate["scopes"]["raw_full_row"]["duplicate_groups"] == 0
    assert duplicate["scopes"]["model_input_row"]["duplicate_groups"] > 0
    assert duplicate["scope"] == "model_input_row"
    assert duplicate["duplicate_kind"] == "model_input_features"
    assert "business_id" not in duplicate["scopes"]
    assert all(
        candidate["fields"] != ["car_name"]
        for candidate in duplicate["business_id_candidates"]
    )
    assert any(
        candidate["fields"] == ["listing_code"]
        and candidate["status"] == "Candidate"
        for candidate in duplicate["business_id_candidates"]
    )
    rules = {issue["rule_id"] for issue in result["issues"]}
    assert "DQ_CROSS_SPLIT_DUPLICATES" not in rules
    assert "DQ_CROSS_SPLIT_FEATURE_COMBINATIONS" in rules
    feature_issue = next(
        issue for issue in result["issues"]
        if issue["rule_id"] == "DQ_CROSS_SPLIT_FEATURE_COMBINATIONS"
    )
    assert feature_issue["severity"] == "Low"
    assert feature_issue["confidence"] == "Medium"
    assert "not proof of duplicate records or entities" in feature_issue["message"]


def test_exact_full_row_duplicates_are_still_reported(tmp_path):
    frame = pd.DataFrame({
        "feature": [1, 1, 2, 2, 3, 3],
        "target": [0, 0, 1, 1, 0, 0],
    })
    csv_path = tmp_path / "duplicates.csv"
    frame.to_csv(csv_path, index=False)

    result = analyze_csv(csv_path, data_flow=_split_flow(["feature"]))

    assert result["duplicate_rows"] == 3
    assert result["duplicate_groups"] == 3
    assert result["duplicate_split_analysis"]["scopes"]["raw_full_row"][
        "duplicate_groups"
    ] == 3
    assert "DQ_DUPLICATE_ROWS" in {issue["rule_id"] for issue in result["issues"]}
    cross_split = result["duplicate_split_analysis"]["scopes"]["raw_full_row"]
    if cross_split["cross_split_duplicate_groups"]:
        issue = next(
            item for item in result["issues"]
            if item["rule_id"] == "DQ_CROSS_SPLIT_DUPLICATES"
        )
        assert issue["duplicate_kind"] == "full_row"
        assert issue["confidence"] == "High"


def test_explicit_business_id_enables_exact_entity_overlap_analysis(tmp_path):
    frame = pd.DataFrame({
        "customer_key": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "feature": range(8),
        "target": [0, 1] * 4,
    })
    csv_path = tmp_path / "customers.csv"
    frame.to_csv(csv_path, index=False)
    flow = _split_flow(["feature"], random_state=5)

    result = analyze_csv(
        csv_path,
        constraints={"business_id_columns": ["customer_key"]},
        data_flow=flow,
    )

    train_indices, test_indices = train_test_split(
        list(frame.index), test_size=0.5, random_state=5
    )
    expected = len(
        set(frame.loc[train_indices, "customer_key"])
        & set(frame.loc[test_indices, "customer_key"])
    )
    duplicate = result["duplicate_split_analysis"]

    assert duplicate["status"] == "Exact"
    assert duplicate["scope"] == "business_id"
    assert duplicate["duplicate_kind"] == "entity_id"
    assert duplicate["fields"] == ["customer_key"]
    assert duplicate["cross_split_duplicate_groups"] == expected
    assert duplicate["scopes"]["business_id"]["status"] == "Exact"
    entity_issue = next(
        issue for issue in result["issues"]
        if issue["rule_id"] == "DQ_CROSS_SPLIT_ENTITY_ID_OVERLAP"
    )
    assert entity_issue["duplicate_kind"] == "entity_id"
    assert entity_issue["confidence"] == "High"


def test_unreconstructable_split_does_not_invent_exact_overlap_counts(tmp_path):
    frame = pd.DataFrame({
        "record_id": ["a", "a", "b", "b", "c", "c"],
        "feature": [1, 1, 2, 2, 3, 3],
        "target": [0, 0, 1, 1, 0, 0],
    })
    csv_path = tmp_path / "unknown_split.csv"
    frame.to_csv(csv_path, index=False)

    result = analyze_csv(
        csv_path,
        constraints={"business_id_columns": ["record_id"]},
        data_flow=_split_flow(["feature"], random_state=None),
    )
    duplicate = result["duplicate_split_analysis"]

    assert duplicate["status"] == "Unable to quantify"
    assert duplicate["cross_split_duplicate_groups"] is None
    assert duplicate["shared_records"] is None
    assert duplicate["scopes"] == {}


def test_camel_case_id_is_candidate_but_name_fields_are_not(tmp_path):
    frame = pd.DataFrame({
        "CustomerID": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
        "car_name": ["sedan"] * 4 + ["suv"] * 4,
        "record_name": [f"row-{index}" for index in range(8)],
        "feature": range(8),
        "target": [0, 1] * 4,
    })
    csv_path = tmp_path / "camel.csv"
    frame.to_csv(csv_path, index=False)

    result = analyze_csv(csv_path, data_flow=_split_flow(["feature"], random_state=5))
    candidates = result["duplicate_split_analysis"]["business_id_candidates"]

    customer = next(item for item in candidates if item["fields"] == ["CustomerID"])
    assert customer["normalized_name"] == "customer_id"
    assert customer["status"] == "Candidate"
    assert all(item["fields"] != ["car_name"] for item in candidates)
    assert all(item["fields"] != ["record_name"] for item in candidates)
