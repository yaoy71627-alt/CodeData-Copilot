"""Batch B positive cases, version semantics and conservative negative controls."""
import json
from pathlib import Path

import nbformat
import pytest

from code_analyzer import analyze_code_with_ast, analyze_notebook
from static_review_rules import (
    compatibility, documentation_split_findings, read_dependencies,
    slice_sample_findings,
)


def rule(code, identifier, dependencies=None):
    return [i for i in analyze_code_with_ast(code, dependencies) if i["rule_id"] == identifier]


@pytest.mark.parametrize("source,identifier,package,boundary", [
    ("from sklearn.cross_validation import train_test_split", "CODE_SKLEARN_LEGACY_API", "scikit-learn", "0.20"),
    ("from sklearn.preprocessing import OneHotEncoder as OHE\nx=OHE(sparse=True)", "CODE_ONEHOT_SPARSE_API", "scikit-learn", "1.4"),
    ("import pandas as pd\ndf=pd.DataFrame()\ndf.append(other)", "CODE_PANDAS_APPEND", "pandas", "2.0"),
])
@pytest.mark.parametrize("mode", ["unknown", "new", "old", "mixed"])
def test_dependency_version_semantics(source, identifier, package, boundary, mode):
    specs = {"unknown": "", "new": f">={boundary}", "old": f"<{boundary}", "mixed": ">=0.1"}
    dependencies = read_dependencies([("requirements.txt", package + specs[mode])])
    issue, = rule(source, identifier, dependencies)
    assert issue["verification_status"] == ("confirmed_error" if mode == "new" else "needs_review")
    assert issue["dependency_status"] == {"unknown": "unknown", "new": "incompatible", "old": "compatible", "mixed": "unknown"}[mode]
    assert issue["line"] >= 1 and issue["source_snippet"] and issue["evidence"]
    if mode == "new":
        assert "未执行" in issue["message"]


def test_alternative_or_conditional_dependencies_never_prove_incompatibility():
    for docs in [
        [("requirements.txt", "scikit-learn==1.4"), ("requirements-old.txt", "scikit-learn<1.0")],
        [("requirements.txt", 'scikit-learn>=1.4; python_version>="3.10"')],
        [("requirements.txt", "scikit-learn>=1.4,<1.0")],
    ]:
        assert compatibility("scikit-learn", "1.4", read_dependencies(docs))[0] == "unknown"


def test_conda_and_pyproject_declarations():
    deps = read_dependencies([("environment.yml", "dependencies:\n  - scikit-learn=1.4.2\n  - pandas>=2.0"),
                              ("pyproject.toml", '[project]\ndependencies=["scikit-learn>=1.4"]')])
    assert compatibility("scikit-learn", "1.4", deps)[0] == "incompatible"
    assert compatibility("pandas", "2.0", deps)[0] == "incompatible"


@pytest.mark.parametrize("code", [
    "class OneHotEncoder: pass\nx=OneHotEncoder(sparse=True)",
    "from custom import OneHotEncoder\nx=OneHotEncoder(sparse=True)",
    "from sklearn.preprocessing import OneHotEncoder\nOneHotEncoder=custom\nx=OneHotEncoder(sparse=True)",
    "from sklearn.preprocessing import OneHotEncoder\ndef f(OneHotEncoder):\n return OneHotEncoder(sparse=True)",
    "from sklearn.preprocessing import OneHotEncoder\nx=OneHotEncoder(sparse_output=True)",
])
def test_encoder_no_name_only_or_shadowed_false_positive(code):
    assert not rule(code, "CODE_ONEHOT_SPARSE_API")


@pytest.mark.parametrize("code", ["xs=[]\nxs.append(1)", "import pandas as pd\ndf=pd.DataFrame()\ndf=[]\ndf.append(1)"])
def test_append_list_is_not_dataframe(code):
    assert not rule(code, "CODE_PANDAS_APPEND")


def test_function_parameter_does_not_inherit_global_dataframe_type():
    assert not rule("import pandas as pd\ndf=pd.DataFrame()\ndef f(df):\n df.append(1)", "CODE_PANDAS_APPEND")


def test_discarded_status_vs_actual_checks():
    issue, = rule("request.status_code == 200", "CODE_UNUSED_STATUS_COMPARISON")
    assert issue["verification_status"] == "confirmed_error" and issue["line"] == 1
    for source in ["assert request.status_code == 200", "if request.status_code == 200:\n process()", "ok=request.status_code==200", "def ok():\n return request.status_code==200"]:
        assert not rule(source, "CODE_UNUSED_STATUS_COMPARISON")


def test_duplicate_check_is_review_and_pair_is_actually_missing():
    source = "X_train, X_test = split()\ntest_columns(X_test)\ntest_columns(X_test)"
    issue, = rule(source, "CODE_DUPLICATE_CHECK_CALL")
    assert issue["verification_status"] == "needs_review" and issue["confidence"] == "Low"
    assert "X_train" in issue["evidence"]
    for code in [source + "\ntest_columns(X_train)", source.replace("test_columns", "print"), source.replace("test_columns(X_test)\ntest_columns(X_test)", "test_columns(X_train)\ntest_columns(X_test)"), "test_columns(X_test)\ntest_columns(X_test)"]:
        assert not rule(code, "CODE_DUPLICATE_CHECK_CALL")


def test_exception_fallthrough_and_safe_paths():
    unsafe = "try:\n response=get()\nexcept Exception:\n print('failed')\nconsume(response)"
    assert rule(unsafe, "CODE_EXCEPTION_UNBOUND_VARIABLE")[0]["verification_status"] == "static_risk"
    for code in ["response=None\n" + unsafe, unsafe.replace("print('failed')", "raise"), unsafe.replace("print('failed')", "response=None"), unsafe.replace("consume(response)", "response=fallback()\nconsume(response)"), unsafe.replace("consume(response)", "finally:\n response=None\nconsume(response)")]:
        assert not rule(code, "CODE_EXCEPTION_UNBOUND_VARIABLE")
    assert not rule("def f(response=None):\n" + "\n".join(" " + line for line in unsafe.splitlines()), "CODE_EXCEPTION_UNBOUND_VARIABLE")
    grouped, = rule("try:\n train_df=read_train()\n test_df=read_test()\nexcept Exception:\n print('failed')\nconsume(train_df,test_df)", "CODE_EXCEPTION_UNBOUND_VARIABLE")
    assert "train_df" in grouped["evidence"] and "test_df" in grouped["evidence"]


def units(code):
    return [{"file": "model.py", "code": code}]


def test_slice_counts_require_verified_row_lineage():
    source = "import pandas as pd\ndf=pd.read_csv('raw.csv')\ndata_selected=df[['a','b']]\ntest_data=data_selected[1469:]"
    result, = slice_sample_findings(units(source), {"raw.csv": 1470})
    issue, = result["issues"]
    assert issue["verified_test_rows"] == 1 and issue["verification_status"] == "needs_review"
    assert "部署连通性烟测" in issue["evidence"]
    for rows in [{}, {"other.csv": 1470}, {"a/raw.csv": 1470, "b/raw.csv": 4000}]:
        unknown, = slice_sample_findings(units(source), rows)
        assert "verified_test_rows" not in unknown["issues"][0]
        assert "不能" in unknown["issues"][0]["message"]


@pytest.mark.parametrize("alteration", [
    "df=df.dropna()", "df.dropna(inplace=True)", "df.loc[9000]=0", "df=arbitrary_transform(df)",
    "if flag:\n df=df[mask]", "del df['a']", "mutate(df)", "df.custom_mutation()", "result=mutate(df)",
])
def test_unknown_or_row_changing_operation_discards_exact_count(alteration):
    source = f"import pandas as pd\ndf=pd.read_csv('raw.csv')\n{alteration}\ntest_data=df[1469:]"
    issue = slice_sample_findings(units(source), {"raw.csv": 1470})[0]["issues"][0]
    assert "verified_test_rows" not in issue


def test_custom_reader_never_inherits_csv_row_count():
    issue = slice_sample_findings(units("from custom import read_csv\ndf=read_csv('raw.csv')\ntest_data=df[1469:]"), {"raw.csv": 1470})[0]["issues"][0]
    assert "verified_test_rows" not in issue


def test_sufficient_slice_and_normal_split_not_small_sample_alerts():
    for source in ["import pandas as pd\ndf=pd.read_csv('raw.csv')\ntest_data=df[1000:]", "X_train, X_test=train_test_split(X,test_size=0.3)"]:
        assert not slice_sample_findings(units(source), {"raw.csv": 1470})


def test_documentation_mismatch_positions_and_ambiguity():
    docs = [{"file": "README.md", "excerpt": "Intro\nSplit (80% train/20% test)."}]
    source = "from sklearn.model_selection import train_test_split\ntrain,test=train_test_split(raw,test_size=0.3)"
    result, = documentation_split_findings(units(source), docs)
    issue, = result["issues"]
    assert issue["line"] == 2 and issue["affected_locations"][0]["line"] == 2
    assert issue["verification_status"] == "needs_review"
    assert not documentation_split_findings(units(source.replace("0.3", "0.2")), docs)
    assert not documentation_split_findings(units(source + "\na,b=train_test_split(other,test_size=ratio)"), docs)
    assert not documentation_split_findings(units(source), [{"file": "README.md", "excerpt": "Accuracy is 80%, test score 20%."}])
    assert not documentation_split_findings(units(source), [{"file": "other/README.md", "excerpt": docs[0]["excerpt"]}])
    assert not documentation_split_findings(units(source), [{"file": "notes.md", "excerpt": docs[0]["excerpt"]}])


@pytest.mark.parametrize("use_alias", [False, True])
def test_notebook_cross_cell_imports_frames_cv_and_locations(tmp_path, use_alias):
    sources = ["import pandas as pd\nfrom sklearn.preprocessing import OneHotEncoder\nfrom sklearn.model_selection import GridSearchCV",
               "df=pd.DataFrame()\nencoder=OneHotEncoder(sparse=True)",
               "df.append(other)\nmodel.fit(X_train,y_train)",
               "search=GridSearchCV(model,grid)\nsearch.fit(X_train,y_train)"]
    if use_alias:
        sources[0] = sources[0].replace("import GridSearchCV", "import GridSearchCV as Search")
        sources[3] = sources[3].replace("GridSearchCV(", "Search(")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources])
    path = tmp_path / "model.ipynb"
    nbformat.write(notebook, path)
    results = analyze_notebook(path)
    all_issues = [(row["cell"], issue) for row in results for issue in row["issues"]]
    assert [(cell, i["line"]) for cell, i in all_issues if i["rule_id"] == "CODE_ONEHOT_SPARSE_API"] == [(2, 2)]
    assert [(cell, i["line"]) for cell, i in all_issues if i["rule_id"] == "CODE_PANDAS_APPEND"] == [(3, 1)]
    assert not [i for _, i in all_issues if i["rule_id"] == "CODE_NO_CROSS_VALIDATION"]


def test_commented_cv_is_not_evidence(tmp_path):
    path = tmp_path / "model.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("model.fit(X,y)\n# GridSearchCV(model,grid)")]), path)
    assert any(i["rule_id"] == "CODE_NO_CROSS_VALIDATION" for row in analyze_notebook(path) for i in row["issues"])


def test_column_mask_writes_and_chart_properties_preserve_verified_rows():
    source = "import pandas as pd\ndf=pd.read_csv('raw.csv')\ndf.loc[(df['a']>1)&(df['a']<5),'b']=0\nchart.labels=range(10)\ntest_data=df[1469:]"
    assert slice_sample_findings(units(source), {"raw.csv": 1470})[0]["issues"][0]["verified_test_rows"] == 1


def test_github_scanner_admits_only_supported_dependency_manifests(tmp_path):
    from github_loader import _scan_repository
    for name in ("model.py", "requirements.txt", "environment.yml", "pyproject.toml", "other.txt"):
        (tmp_path / name).write_text("# test", encoding="utf-8")
    files = _scan_repository(tmp_path)
    assert {name for name, _ in files} == {"model.py", "requirements.txt", "environment.yml", "pyproject.toml"}
    assert any(item["path"] == "other.txt" for item in files.skipped_files)
