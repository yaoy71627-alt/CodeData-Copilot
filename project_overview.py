"""Lightweight static project understanding for the overview dashboard."""

import ast
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


LIBRARY_LABELS = {
    "pandas": "pandas",
    "numpy": "NumPy",
    "sklearn": "scikit-learn",
    "matplotlib": "Matplotlib",
    "seaborn": "seaborn",
    "scipy": "SciPy",
    "statsmodels": "statsmodels",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "geopandas": "GeoPandas",
    "shapely": "Shapely",
    "qgis": "QGIS",
    "plotly": "Plotly",
    "streamlit": "Streamlit",
    "polars": "Polars",
}


def _source_signals(sources):
    imports, calls = set(), set()
    for source in sources:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                function = node.func
                name = (function.attr if isinstance(function, ast.Attribute)
                        else function.id if isinstance(function, ast.Name) else None)
                if name:
                    calls.add(name)
    return imports, calls


def _project_name(file_names, metadata):
    repository_url = (metadata or {}).get("repository_url", "")
    if repository_url:
        name = Path(urlsplit(repository_url).path.rstrip("/")).name.removesuffix(".git")
        if name:
            return name
    explicit = (metadata or {}).get("project_name")
    if explicit:
        return explicit
    code_files = [Path(name) for name in file_names if Path(name).suffix.lower() in {".py", ".ipynb"}]
    if len(code_files) == 1:
        return code_files[0].stem
    return "Local Data Science Project"


def _project_structure(imports, calls):
    structure = []
    if imports & {"pandas", "polars", "numpy", "geopandas"} or calls & {
            "read_csv", "read_excel", "read_parquet", "fillna", "dropna", "merge"}:
        structure.append("数据读取与处理")
    if calls & {"fit_transform", "transform", "get_dummies", "OneHotEncoder",
                "LabelEncoder", "SelectKBest", "PolynomialFeatures"}:
        structure.append("特征工程")
    if calls & {"fit", "partial_fit", "train"} or imports & {
            "sklearn", "torch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost"}:
        structure.append("模型构建与训练")
    if calls & {"predict", "score", "classification_report", "confusion_matrix",
                "accuracy_score", "mean_squared_error", "roc_auc_score", "cross_val_score"}:
        structure.append("模型评价")
    if imports & {"matplotlib", "seaborn", "plotly"} or calls & {"plot", "scatter", "hist", "imshow"}:
        structure.append("可视化分析")
    if imports & {"geopandas", "shapely", "qgis"}:
        structure.append("空间数据分析")
    return structure or ["通用 Python 模块"]


def _project_type(imports, structure, data_file_count):
    if imports & {"geopandas", "shapely", "qgis"}:
        return "Geospatial Data Science Project"
    if imports & {"sklearn", "torch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost"}:
        return "Machine Learning Project"
    if data_file_count or "数据读取与处理" in structure:
        return "Data Science Project"
    return "Python Analytics Project"


def build_project_overview(file_names, sources, data_results, metadata=None):
    """Return project-level facts inferred from files and static Python sources."""
    file_names = list(file_names or [])
    imports, calls = _source_signals(sources or [])
    suffix_counts = Counter(Path(name).suffix.lower() or "[no extension]" for name in file_names)
    python_count = suffix_counts[".py"] + suffix_counts[".ipynb"]
    data_file_count = suffix_counts[".csv"]

    languages = []
    if python_count:
        languages.append("Python")
    if suffix_counts[".ipynb"]:
        languages.append("Jupyter Notebook")
    if suffix_counts[".md"]:
        languages.append("Markdown")

    libraries = [label for name, label in LIBRARY_LABELS.items() if name in imports]
    structure = _project_structure(imports, calls)
    project_type = _project_type(imports, structure, data_file_count)
    data_files = []
    for entry in data_results or []:
        report = entry.get("result", entry)
        shape = report.get("shape", {})
        data_files.append({
            "file": entry.get("file", "CSV"),
            "rows": shape.get("rows", 0),
            "columns": shape.get("columns", 0),
            "missing_columns": len(report.get("missing_values", {})),
            "duplicate_rows": report.get("duplicate_rows", 0),
        })

    name = _project_name(file_names, metadata or {})
    library_text = "、".join(libraries[:6]) if libraries else "未识别到常见数据科学库"
    summary = (
        f"{name} 被识别为 {project_type}，共纳入 {len(file_names)} 个文件，"
        f"其中 {python_count} 个 Python/Notebook 文件、{data_file_count} 个数据文件。"
        f"静态代码显示项目主要使用 {library_text}，涉及{'、'.join(structure)}。"
    )
    if data_files:
        summary += f" 数据分析覆盖 {sum(item['rows'] for item in data_files)} 行记录。"

    return {
        "project_name": name,
        "project_type": project_type,
        "file_count": len(file_names),
        "python_file_count": python_count,
        "data_file_count": data_file_count,
        "documentation_file_count": suffix_counts[".md"],
        "languages": languages or ["未识别"],
        "main_libraries": libraries,
        "project_structure": structure,
        "file_types": dict(sorted(suffix_counts.items())),
        "data_files": data_files,
        "summary": summary,
    }
