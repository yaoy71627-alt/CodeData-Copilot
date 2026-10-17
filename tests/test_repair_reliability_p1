"""Conservative repair validation and exact, read-only whole-file patch export."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

import agent
from code_analyzer import analyze_code_with_ast
from data_flow_analyzer import analyze_python_data_flow
from repair_validator import VALIDATION_STATUSES, validate_repair_suggestions
from patch_generator import apply_verified_diff, build_unified_patch


def repair(source="x = 1\nvalue = x + 1\nprint(value)\n", before="value = x + 1", after="value = x + 2", line=2, file="demo.py", **extra):
    flow = analyze_python_data_flow(source, file_name=file)
    suggestion = {"file": file, "location": file, "line": line,
                  "before_code": before, "after_code": after, **extra}
    return flow, suggestion


def validate(flow, suggestion, data=None):
    return validate_repair_suggestions([suggestion], flow, data or [])[0]


def test_safe_static_patch_uses_whole_file_line_numbers_and_applies_exactly():
    source = "x = 1\n" + "# context\n" * 12 + "value = x + 1\nprint(value)\n"
    flow, item = repair(source=source, line=14)
    result = validate(flow, item)
    assert result["validation_status"] == "Statically Validated"
    assert not result["runtime_verified"]
    diff = build_unified_patch([result])
    assert "@@ -11," in diff
    assert apply_verified_diff(source, diff) == source.replace("value = x + 1", "value = x + 2")


@pytest.mark.parametrize("after", ["value = ghost(x)", "value = pd.DataFrame()", "value = np.mean(x)", "value = not_defined + 1"])
def test_unknown_functions_aliases_and_variables_never_exposed_as_executable(after):
    flow, item = repair(after=after)
    result = validate(flow, item)
    assert result["validation_status"] == "Failed Validation"
    assert result["after_code"] is None and not result["executable"]
    assert not build_unified_patch([result])


@pytest.mark.parametrize("after", ['value = df["Ghost"]', 'value = df[["Age", "Ghost"]]',
                                   'value = df.loc[:, ["Ghost"]]', 'value = df.Ghost',
                                   'value = df.drop(columns=["Ghost"])'])
def test_all_common_column_selectors_reject_unknown_fields(after):
    source = 'import pandas as pd\ndf = pd.read_csv("train.csv")\nvalue = df["Age"]\n'
    flow, item = repair(source, 'value = df["Age"]', after, 3)
    data = [{"file": "train.csv", "result": {"data_types": {"Age": "float64"}}}]
    result = validate(flow, item, data)
    assert result["validation_status"] == "Failed Validation"
    assert "Ghost" in result["validation_detail"]
    assert not result["after_code"]


def test_unrelated_dataset_cannot_supply_missing_column():
    source = 'import pandas as pd\ndf = pd.read_csv("train.csv")\nvalue = df["Age"]\n'
    flow, item = repair(source, 'value = df["Age"]', 'value = df["Cabin"]', 3)
    data = [{"file": "train.csv", "result": {"data_types": {"Age": "float64"}}},
            {"file": "other.csv", "result": {"data_types": {"Cabin": "object"}}}]
    result = validate(flow, item, data)
    assert result["validation_status"] == "Failed Validation"
    assert "Cabin" in result["validation_detail"]


def test_cabin_existing_in_csv_but_excluded_from_model_is_rejected():
    source = 'import pandas as pd\ndf = pd.read_csv("train.csv")\nX = df[["Age"]]\nvalue = X["Age"]\n'
    flow, item = repair(source, 'value = X["Age"]', 'value = X["Cabin"]', 4, scope="model_input")
    result = validate(flow, item, [{"file": "train.csv", "result": {"data_types": {"Age": "float64", "Cabin": "object"}}}])
    assert result["validation_status"] == "Failed Validation"
    assert not result["after_code"]


def test_missing_schema_is_syntax_only_and_hides_candidate():
    source = 'import pandas as pd\ndf = pd.read_csv("train.csv")\nvalue = 1\n'
    flow, item = repair(source, "value = 1", 'value = df["Age"]', 3)
    result = validate(flow, item)
    assert result["validation_status"] == "Syntax Valid Only"
    assert result["candidate_code"] == 'value = df["Age"]'
    assert not result["after_code"] and not build_unified_patch([result])


def test_unknown_api_remains_unverified_even_if_variable_exists():
    flow, item = repair(source="x = 1\nvalue = x + 1\n", after="value = x.magic()")
    result = validate(flow, item)
    assert result["validation_status"] == "API Unverified"
    assert not result["after_code"] and not build_unified_patch([result])


def test_missing_position_or_wrong_line_never_exports_patch():
    flow, item = repair()
    item["line"] = None
    result = validate(flow, item)
    assert result["validation_status"] == "Syntax Valid Only"
    assert not build_unified_patch([result])
    item["line"] = 1
    result = validate(flow, item)
    assert not build_unified_patch([result])
    assert "不一致" in result["validation_detail"]


def test_duplicate_text_is_replaced_at_declared_location_only():
    source = "x = 1\nvalue = x + 1\nvalue = x + 1\nprint(value)\n"
    flow, item = repair(source, line=3)
    result = validate(flow, item)
    diff = build_unified_patch([result])
    assert apply_verified_diff(source, diff) == "x = 1\nvalue = x + 1\nvalue = x + 2\nprint(value)\n"


def test_deleted_definition_used_downstream_is_rejected():
    flow, item = repair(after="other = x + 2")
    result = validate(flow, item)
    assert result["validation_status"] == "Failed Validation"
    assert "value" in result["validation_detail"]


def test_function_local_is_not_visible_in_top_level_repair():
    source = "def helper():\n    hidden = 1\nx = 1\nvalue = x + 1\n"
    flow, item = repair(source, after="value = hidden + 1", line=4)
    result = validate(flow, item)
    assert result["validation_status"] == "Failed Validation"
    assert "hidden" in result["validation_detail"]


def test_leakage_and_missing_value_repairs_are_directions_only():
    source = '''features = ["Age", "Fare"]
X = df[features]
y = df["Survived"]
model_data = df[features + ["Survived"]]
model_data = model_data.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
'''
    flow = analyze_python_data_flow(source, "titanic.py")
    issues = analyze_code_with_ast(source)
    data = [{"file": "train.csv", "result": {"data_types": {"Age": "float64", "Fare": "float64", "Cabin": "object", "Survived": "int64"}, "missing_values": {"Age": 0.2, "Cabin": 0.8}}}]
    suggestions = agent.build_repair_suggestions([{"file": "titanic.py", "issues": issues}], flow, data)
    validated = validate_repair_suggestions(suggestions, flow, data)
    relevant = [i for i in validated if i["problem"] in {"Data Leakage Risk", "Potential Data Leakage", "Missing Value Handling Risk"}]
    assert relevant
    for item in relevant:
        assert item["validation_status"] == "Recommendation Only"
        assert item["after_code"] is None and item["conceptual_code"] is None
        assert "Cabin" not in item.get("referenced_columns", [])
    assert not build_unified_patch(relevant)


def test_coordinated_preprocessing_cannot_be_forged_as_single_line_fix():
    source = "X = 1\nscaler = StandardScaler()\nscaled = scaler.fit_transform(X)\n"
    flow, item = repair(source, "scaled = scaler.fit_transform(X)", "scaled = scaler.transform(X)", 3,
                        problem="Data Leakage Risk")
    result = validate(flow, item)
    assert result["validation_status"] == "Recommendation Only"
    assert not build_unified_patch([result])


def test_patch_requires_real_verification_receipt_not_status_label():
    for status in [None, "Statically Validated", "Runtime Verified", "Syntax Valid Only"]:
        item = {"file": "demo.py", "line": 1, "before_code": "x=1", "after_code": "x=2", "validation_status": status}
        assert not build_unified_patch([item])


def test_forged_runtime_label_is_never_accepted_as_runtime_execution():
    flow, item = repair(validation_status="Runtime Verified", runtime_verified=True)
    result = validate(flow, item)
    assert result["validation_status"] == "Statically Validated"
    assert result["runtime_verified"] is False
    assert result["validation_checks"]["runtime_executed"] is False


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_patch_roundtrip_preserves_file_endings(ending):
    source = "x = 1" + ("\r\n" if ending == "\r\n" else "\n") + "value = x + 1" + ending
    flow, item = repair(source)
    result = validate(flow, item)
    diff = build_unified_patch([result])
    assert diff
    assert apply_verified_diff(source, diff) == source.replace("value = x + 1", "value = x + 2")


def test_two_nonoverlapping_repairs_export_one_combined_file_diff():
    source = "x = 1\na = x + 1\nb = x + 1\nprint(a, b)\n"
    flow, first = repair(source, "a = x + 1", "a = x + 2", 2)
    _, second = repair(source, "b = x + 1", "b = x + 3", 3)
    checked = validate_repair_suggestions([first, second], flow, [])
    diff = build_unified_patch(checked)
    assert diff.count("--- a/demo.py") == 1
    assert apply_verified_diff(source, diff) == source.replace("a = x + 1", "a = x + 2").replace("b = x + 1", "b = x + 3")


def test_conflicting_repairs_for_same_span_are_not_exported():
    flow, first = repair()
    _, second = repair(after="value = x + 3")
    assert not build_unified_patch(validate_repair_suggestions([first, second], flow, []))


def test_notebook_context_never_exported_as_if_it_were_notebook_json():
    flow, item = repair(file="demo.ipynb", cell=2)
    flow["code_units"][0]["cell"] = 2
    for locations in flow["definitions"].values():
        for location in locations:
            location["cell"] = 2
    result = validate(flow, item)
    assert result["validation_status"] == "Statically Validated"
    assert result["patch_validation"]["scope"] == "notebook_cell_fragment"
    assert result["patch_validation"]["project_applicable"] is False
    assert not build_unified_patch([result])


def test_invalid_diff_cannot_be_applied_to_other_source():
    flow, item = repair()
    diff = build_unified_patch([validate(flow, item)])
    with pytest.raises(ValueError):
        apply_verified_diff("something else\n", diff)


def test_validation_and_generation_leave_inputs_and_uploaded_files_unchanged(tmp_path):
    source = "x = 1\nvalue = x + 1\nprint(value)\n"
    path = tmp_path / "demo.py"; path.write_text(source)
    flow, item = repair(source)
    saved_flow, saved_item = copy.deepcopy(flow), copy.deepcopy(item)
    before = path.read_bytes()
    result = validate(flow, item); build_unified_patch([result])
    assert path.read_bytes() == before
    assert flow == saved_flow and item == saved_item


def test_all_six_statuses_remain_defined_without_fabricating_runtime_proof():
    assert VALIDATION_STATUSES == {"Recommendation Only", "Syntax Valid Only", "API Unverified", "Statically Validated", "Runtime Verified", "Failed Validation"}


def test_import_after_use_is_not_considered_an_available_alias():
    flow, item = repair(after="value = pd.DataFrame()\nimport pandas as pd")
    result = validate(flow, item)
    assert result["validation_status"] == "Failed Validation"
    assert "pd" in result["validation_detail"]


def test_local_dataframe_selection_cannot_borrow_excluded_columns():
    source = 'import pandas as pd\ndf = pd.read_csv("train.csv")\nvalue = 1\n'
    flow, item = repair(source, "value = 1", 'view = df[["Age"]]\nvalue = view["Cabin"]', 3)
    data = [{"file": "train.csv", "result": {"data_types": {"Age": "float64", "Cabin": "object"}}}]
    result = validate(flow, item, data)
    assert result["validation_status"] == "Failed Validation"
    assert "Cabin" in result["validation_detail"]


def test_revalidation_preserves_limited_status_and_never_promotes_empty_code():
    flow, item = repair(after="value = x.magic()")
    first = validate(flow, item)
    second = validate(flow, first)
    assert first["validation_status"] == second["validation_status"] == "API Unverified"
    assert first["candidate_code"] == second["candidate_code"]
    assert not build_unified_patch([second])
    forged = {"validation_status": "Runtime Verified", "runtime_verified": True}
    assert validate(flow, forged)["validation_status"] == "Recommendation Only"


def test_tampering_code_after_validation_invalidates_export_receipt():
    flow, item = repair()
    checked = validate(flow, item)
    assert build_unified_patch([checked])
    checked["after_code"] = "value = not_defined + 1"
    assert not build_unified_patch([checked])


def test_report_final_boundary_discards_cached_or_injected_patch():
    from consistency_validator import validate_report_consistency
    checked = validate_report_consistency({"analysis_facts": {}, "ai_review": "基础诊断。",
        "code_quality": {"issues": []}, "patch_text": "unverified cached diff",
        "repair_suggestions": [{"before_code": "x=1", "after_code": "x=2",
                                 "validation_status": "Statically Validated"}]})
    assert checked["patch_text"] == ""


def test_patch_path_is_not_silently_rewritten_to_other_file():
    for name in ["../demo.py", "/tmp/demo.py", "C:\\work\\demo.py", "some name.py"]:
        flow, item = repair(file=name)
        checked = validate(flow, item)
        assert not build_unified_patch([checked])


def test_patch_parser_rejects_modified_hunk_counts():
    flow, item = repair()
    diff = build_unified_patch([validate(flow, item)])
    damaged = diff.replace("@@ -1,3 +1,3 @@", "@@ -1,9 +1,3 @@")
    assert damaged != diff
    with pytest.raises(ValueError):
        apply_verified_diff(flow["code_units"][0]["code"], damaged)


def test_astype_mapping_cannot_reference_unknown_field():
    source = 'import pandas as pd\ndf = pd.read_csv("train.csv")\nvalue = 1\n'
    flow, item = repair(source, "value = 1", 'value = df.astype({"Ghost": float})', 3)
    result = validate(flow, item, [{"file": "train.csv", "result": {"data_types": {"Age": "float64"}}}])
    assert result["validation_status"] == "Failed Validation"
    assert "Ghost" in result["validation_detail"]


def test_removing_import_alias_used_downstream_is_not_a_safe_patch():
    source = "import pandas as pd\nvalue = pd.DataFrame()\n"
    flow, item = repair(source, "import pandas as pd", "import pandas as other", 1)
    result = validate(flow, item)
    assert result["validation_status"] == "Failed Validation"
    assert "pd" in result["validation_detail"]
    assert not build_unified_patch([result])


def test_report_page_refreshes_legacy_cached_exports_before_download():
    from test_app import build_app
    at = build_app("Report", with_results=True)
    assert not at.exception
    at.session_state["report_data"] = {"repair_validation_policy": "old"}
    at.session_state["report_markdown"] = "cached unsafe patch text"
    at.session_state["patch_text"] = "cached unsafe patch text"
    at.run()
    assert not at.exception
    assert at.session_state["report_data"]["repair_validation_policy"] == "exact-source-p1"
    assert "cached unsafe patch text" not in at.session_state["report_markdown"]
    assert not at.session_state["report_data"]["patch_text"]
