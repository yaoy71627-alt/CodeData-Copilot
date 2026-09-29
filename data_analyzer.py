"""Evidence-based data quality analysis for tabular data science projects."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, defaultdict

import pandas as pd


VERY_SMALL_ROW_THRESHOLD = 20
MIN_OUTLIER_SAMPLES = 20
NEAR_CONSTANT_THRESHOLD = 0.99
HIGH_CORRELATION_THRESHOLD = 0.95
HIGH_CARDINALITY_THRESHOLD = 0.90
POTENTIAL_ID_THRESHOLD = 0.98
RARE_CATEGORY_THRESHOLD = 0.01
MIN_RARE_CATEGORY_ROWS = 100
PARSE_SUCCESS_THRESHOLD = 0.80
MAX_IDENTICAL_COLUMNS = 200
MAX_CORRELATION_COLUMNS = 100
MAX_SAMPLE_VALUES = 5
EXCESSIVE_TEXT_LENGTH = 10_000

MISSING_TOKENS = {"na", "n/a", "null", "none", "?", "-", "--", "unknown"}
TARGET_NAMES = {"target", "label", "class", "outcome", "y"}
BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n", "1", "0"}
DATE_TOKEN_RE = re.compile(r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")
NON_NEGATIVE_NAME_RE = re.compile(
    r"(?:^|_)(?:selling_?price|price|fare|cost|amount|count|quantity|distance|mileage)(?:$|_)",
    re.I,
)


def _clean_scalar(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, 6)
    if hasattr(value, "item"):
        try:
            return _clean_scalar(value.item())
        except (TypeError, ValueError):
            pass
    text = str(value)
    return text if len(text) <= 160 else text[:157] + "…"


def _samples(values, limit=MAX_SAMPLE_VALUES):
    output = []
    for value in values:
        cleaned = _clean_scalar(value)
        if cleaned not in output:
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _numeric_variable_profile(column, finite):
    """Classify numeric semantics before choosing an outlier method."""
    unique_count = int(finite.nunique()) if len(finite) else 0
    unique_ratio = unique_count / max(1, len(finite))
    integer_like = bool(len(finite)) and bool(((finite - finite.round()).abs() < 1e-9).all())
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).casefold())
    count_name = bool(re.search(r"(?:count|times?|number|num|visits?|years?|months?)", normalized))
    values = set(float(value) for value in finite.unique())
    if unique_count <= 2 and values.issubset({0.0, 1.0}):
        variable_type = "binary"
    elif integer_like and finite.min() >= 0 and unique_count <= 20 and (
        count_name or unique_ratio <= 0.10
    ):
        variable_type = "count-like"
    elif unique_count <= 10 and unique_ratio <= 0.05:
        variable_type = "low-cardinality numeric categorical"
    elif integer_like and unique_count <= 20 and unique_ratio <= 0.10:
        variable_type = "discrete integer"
    else:
        variable_type = "continuous numeric"
    return {
        "type": variable_type,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "integer_like": integer_like,
        "outlier_method": "IQR/MAD eligible" if variable_type == "continuous numeric" else "Distribution review only",
    }


def _issue(
    rule_id,
    category,
    issue_type,
    severity,
    confidence,
    message,
    recommendation,
    *,
    column=None,
    evidence=None,
    count=None,
    rate=None,
    sample_values=None,
    issue_origin=None,
    issue_nature=None,
    action=None,
    affected_rows=None,
    **extra,
):
    """Create the stable issue schema while retaining report/UI compatibility."""
    if issue_origin is None:
        if category in {"Outliers", "Numeric Quality", "Correlation"}:
            issue_origin = "Statistical analysis"
        elif category in {"Categorical Quality", "Target Quality", "Feature Quality"}:
            issue_origin = "Heuristic profile"
        else:
            issue_origin = "Observed data"
    if issue_nature is None:
        if category == "Missing Values":
            issue_nature = "Missing Data"
        elif rule_id in {"DQ_NON_FINITE_NUMERIC"}:
            issue_nature = "Confirmed Invalid"
        elif rule_id == "DQ_CONSTRAINT_VIOLATION":
            issue_nature = "Rule Violation"
        elif category in {"Type Consistency", "String Quality"}:
            issue_nature = "Format Inconsistency"
        elif rule_id == "DQ_RARE_CATEGORIES":
            issue_nature = "Rare but Valid"
        elif rule_id in {"DQ_POTENTIAL_ID", "DQ_POTENTIAL_INDEX_COLUMN"}:
            issue_nature = "Potential Identifier"
        elif category in {"Numeric Quality", "Correlation"}:
            issue_nature = "Distribution Characteristic"
        elif category == "Outliers":
            issue_nature = "Statistical Anomaly"
        else:
            issue_nature = "Needs Review"
    if action is None:
        if rule_id in {
            "DQ_NON_FINITE_NUMERIC", "DQ_CONSTRAINT_VIOLATION",
            "DQ_NUMERIC_PARSE_ANOMALY", "DQ_DATETIME_PARSE_ANOMALY",
        }:
            action = "Fix"
        elif rule_id in {
            "DQ_RARE_CATEGORIES", "DQ_EXTREME_SKEWNESS", "DQ_VERY_SMALL_DATASET",
        }:
            action = "Informational"
        else:
            action = "Review"
    item = {
        "rule_id": rule_id,
        "category": category,
        "type": issue_type,
        "severity": severity,
        "confidence": confidence,
        "column": column,
        "message": message,
        "evidence": evidence or message,
        "count": int(count) if count is not None else None,
        "rate": float(rate) if rate is not None else None,
        "sample_values": list(sample_values or []),
        "recommendation": recommendation,
        "issue_origin": issue_origin,
        "issue_nature": issue_nature,
        "action": action,
        "affected_rows": list(affected_rows or []),
    }
    item.update(extra)
    return item


def _read_header(csv_path, encoding):
    try:
        with open(csv_path, "r", encoding=encoding, newline="") as handle:
            return next(csv.reader(handle), [])
    except (OSError, StopIteration, csv.Error):
        return []


def _read_csv(csv_path):
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            header = _read_header(csv_path, encoding)
            return pd.read_csv(
                csv_path,
                encoding=encoding,
                keep_default_na=False,
                na_values=[""],
            ), header
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), header
    raise ValueError("无法识别 CSV 编码，请转换为 UTF-8 或 GB18030 后重试。")


def read_csv_for_analysis(csv_path):
    """Return the same parsing view used by quality analysis for safe quantification."""
    return _read_csv(csv_path)[0]


def _severity_for_rate(rate, medium=0.10, high=0.50):
    if rate >= high:
        return "High"
    if rate >= medium:
        return "Medium"
    return "Low"


def _numeric_statistics(series):
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[numeric.map(lambda value: pd.notna(value) and math.isfinite(float(value)))]
    if finite.empty:
        return {
            "count": 0, "missing": int(series.isna().sum()), "mean": None,
            "median": None, "std": None, "min": None, "max": None,
            "q1": None, "q3": None, "skewness": None, "unique_count": 0,
        }
    return {
        "count": int(finite.count()),
        "missing": int(series.isna().sum()),
        "mean": _clean_scalar(finite.mean()),
        "median": _clean_scalar(finite.median()),
        "std": _clean_scalar(finite.std()),
        "min": _clean_scalar(finite.min()),
        "max": _clean_scalar(finite.max()),
        "q1": _clean_scalar(finite.quantile(0.25)),
        "q3": _clean_scalar(finite.quantile(0.75)),
        "skewness": _clean_scalar(finite.skew()),
        "unique_count": int(finite.nunique(dropna=True)),
    }


def _detect_identical_columns(df, issues):
    if df.shape[1] < 2 or df.shape[1] > MAX_IDENTICAL_COLUMNS:
        return
    signatures = defaultdict(list)
    for column in df.columns:
        try:
            hashed = pd.util.hash_pandas_object(df[column], index=False).values.tobytes()
            signature = hashlib.sha256(hashed).hexdigest()
            signatures[(str(df[column].dtype), signature)].append(column)
        except (TypeError, ValueError):
            continue
    for candidates in signatures.values():
        if len(candidates) < 2:
            continue
        reference = candidates[0]
        for candidate in candidates[1:]:
            if df[reference].equals(df[candidate]):
                issues.append(_issue(
                    "DQ_IDENTICAL_COLUMNS", "Feature Quality", "Identical Columns",
                    "Medium", "High",
                    f"字段 {reference} 与 {candidate} 的内容完全相同。",
                    "确认两个字段是否语义重复；不要在未理解业务含义前自动删除。",
                    column=f"{reference} / {candidate}",
                    evidence=f"逐行哈希与等值复核均一致，共 {len(df)} 行。",
                    count=len(df), rate=1.0,
                ))


def _detect_string_and_type_quality(df, issues, profiles):
    row_count = len(df)
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty or pd.api.types.is_numeric_dtype(series):
            continue

        type_families = {type(value).__name__ for value in non_null}
        if len(type_families) > 1:
            issues.append(_issue(
                "DQ_MIXED_PYTHON_TYPES", "Type Consistency", "Mixed Type Column",
                "Medium", "High",
                f"字段 {column} 同时包含多种运行时数据类型。",
                "统一字段类型，并在转换失败时保留原始值供审查。",
                column=column, evidence=f"检测到类型：{', '.join(sorted(type_families))}",
                count=len(type_families), sample_values=_samples(non_null),
            ))

        text = non_null.astype(str)
        stripped = text.str.strip()
        meaningful = stripped[
            (stripped != "") & (~stripped.str.casefold().isin(MISSING_TOKENS))
        ]
        if not meaningful.empty:
            numeric_parsed = pd.to_numeric(meaningful, errors="coerce")
            numeric_ratio = float(numeric_parsed.notna().mean())
            failed = meaningful[numeric_parsed.isna()]
            if PARSE_SUCCESS_THRESHOLD <= numeric_ratio < 1 and not failed.empty:
                issues.append(_issue(
                    "DQ_NUMERIC_PARSE_ANOMALY", "Type Consistency", "Numeric Parsing Anomaly",
                    "Medium", "High",
                    f"字段 {column} 大部分值可解析为数值，但仍有少量失败值。",
                    "核对失败值的来源并采用显式、可审计的数值转换。",
                    column=column,
                    evidence=f"{numeric_ratio:.2%} 可解析；{len(failed)} 个值解析失败。",
                    count=len(failed), rate=len(failed) / max(1, len(meaningful)),
                    sample_values=_samples(failed),
                ))
            elif 0.20 <= numeric_ratio < PARSE_SUCCESS_THRESHOLD:
                parsed_count = int(numeric_parsed.notna().sum())
                failed_count = int(numeric_parsed.isna().sum())
                if parsed_count >= 3 and failed_count >= 3:
                    issues.append(_issue(
                        "DQ_MIXED_REPRESENTATION", "Type Consistency", "Mixed Type Column",
                        "Medium", "Medium",
                        f"字段 {column} 同时包含数值表示和非数值文本。",
                        "确认字段语义；若应为数值，请审计转换失败值，若为类别，请统一编码约定。",
                        column=column,
                        evidence=(
                            f"{parsed_count} 个值可解析为数值，{failed_count} 个值不可解析；"
                            "该结论为表示层诊断。"
                        ),
                        count=failed_count, rate=failed_count / max(1, len(meaningful)),
                        sample_values=_samples(failed),
                    ))

        date_candidates = meaningful[meaningful.str.contains(DATE_TOKEN_RE, regex=True, na=False)]
        if len(date_candidates) >= max(3, math.ceil(len(meaningful) * 0.8)):
            parsed_dates = pd.to_datetime(meaningful, errors="coerce", format="mixed")
            date_ratio = float(parsed_dates.notna().mean())
            failed_dates = meaningful[parsed_dates.isna()]
            if PARSE_SUCCESS_THRESHOLD <= date_ratio < 1 and not failed_dates.empty:
                issues.append(_issue(
                    "DQ_DATETIME_PARSE_ANOMALY", "Type Consistency", "Datetime Parsing Anomaly",
                    "Medium", "High",
                    f"字段 {column} 大部分值符合日期格式，但存在无法解析的值。",
                    "统一日期格式并显式记录解析失败值。",
                    column=column,
                    evidence=f"{date_ratio:.2%} 可解析；{len(failed_dates)} 个值解析失败。",
                    count=len(failed_dates), rate=len(failed_dates) / max(1, len(meaningful)),
                    sample_values=_samples(failed_dates),
                ))

        normalized_boolean = stripped.str.casefold()
        boolean_values = set(normalized_boolean.unique())
        if boolean_values and boolean_values.issubset(BOOLEAN_TOKENS):
            uses_numeric = bool(boolean_values & {"0", "1"})
            uses_words = bool(boolean_values - {"0", "1"})
            if uses_numeric and uses_words:
                issues.append(_issue(
                    "DQ_BOOLEAN_ENCODING", "Type Consistency",
                    "Potential Boolean Encoding Inconsistency", "Low", "Medium",
                    f"字段 {column} 混用了数值和文本布尔编码。",
                    "在建模前统一布尔编码，并保留映射说明。",
                    column=column, evidence=f"检测到编码：{', '.join(sorted(boolean_values))}",
                    count=len(boolean_values), sample_values=_samples(stripped),
                ))

        whitespace_mask = text != stripped
        if whitespace_mask.any():
            affected = text[whitespace_mask]
            issues.append(_issue(
                "DQ_SURROUNDING_WHITESPACE", "String Quality",
                "Leading / Trailing Whitespace", "Low", "High",
                f"字段 {column} 包含首尾空白字符。",
                "在保留原始数据的前提下进行 trim 标准化。",
                column=column, evidence=f"{int(whitespace_mask.sum())} 个值包含首尾空白。",
                count=int(whitespace_mask.sum()), rate=float(whitespace_mask.mean()),
                sample_values=_samples(affected),
            ))

        case_groups = defaultdict(set)
        for value in stripped.unique()[:5000]:
            case_groups[value.casefold()].add(value)
        inconsistent = [values for values in case_groups.values() if len(values) > 1]
        if inconsistent:
            examples = [" / ".join(sorted(values)[:3]) for values in inconsistent[:3]]
            issues.append(_issue(
                "DQ_CATEGORY_CASE", "String Quality", "Category Normalization Issue",
                "Low", "High",
                f"字段 {column} 中存在仅大小写不同的类别值。",
                "在确认类别语义后统一大小写与空白规范。",
                column=column, evidence="；".join(examples),
                count=len(inconsistent), sample_values=examples,
            ))

        long_mask = text.str.len() > EXCESSIVE_TEXT_LENGTH
        if long_mask.any():
            issues.append(_issue(
                "DQ_EXCESSIVE_TEXT", "String Quality", "Excessively Long Text",
                "Low", "High",
                f"字段 {column} 存在明显超长文本值。",
                "确认该字段是否包含误拼接内容、日志或非预期文档。",
                column=column, evidence=f"最大长度 {int(text.str.len().max())} 字符。",
                count=int(long_mask.sum()), rate=float(long_mask.mean()),
                sample_values=[f"length={len(value)}" for value in text[long_mask].head(5)],
            ))

        unique_count = int(stripped.nunique(dropna=True))
        unique_ratio = unique_count / max(1, row_count)
        counts = stripped.value_counts(dropna=True)
        profiles[column] = {
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "top_values": {str(key): int(value) for key, value in counts.head(10).items()},
            "boolean_tokens": sorted(boolean_values) if boolean_values.issubset(BOOLEAN_TOKENS) else [],
        }

        if unique_count == 1:
            issues.append(_issue(
                "DQ_SINGLE_CATEGORY", "Categorical Quality",
                "Single-value Categorical Column", "Low", "High",
                f"类别字段 {column} 只有一个有效取值。",
                "确认该字段是否提供建模信息；不要在未确认用途前自动删除。",
                column=column, evidence=f"唯一值：{_clean_scalar(counts.index[0])}",
                count=int(non_null.count()), rate=1.0, sample_values=_samples(counts.index),
            ))
        if row_count >= VERY_SMALL_ROW_THRESHOLD and unique_ratio >= HIGH_CARDINALITY_THRESHOLD:
            issues.append(_issue(
                "DQ_HIGH_CARDINALITY", "Categorical Quality", "High Cardinality",
                "Low", "Medium",
                f"字段 {column} 的唯一值比例较高。",
                "确认该字段是标识符、自由文本还是可泛化的类别特征。",
                column=column, evidence=f"{unique_count}/{row_count} 个唯一值（{unique_ratio:.2%}）。",
                count=unique_count, rate=unique_ratio, sample_values=_samples(stripped),
            ))
        normalized_name = str(column).strip().casefold()
        name_suggests_id = bool(re.search(r"(?:^|_)(?:id|uuid|key|code)$", normalized_name))
        if row_count >= VERY_SMALL_ROW_THRESHOLD and unique_ratio >= POTENTIAL_ID_THRESHOLD and (
            not pd.api.types.is_numeric_dtype(series) or name_suggests_id
        ):
            issues.append(_issue(
                "DQ_POTENTIAL_ID", "Feature Quality", "Potential ID-like Feature",
                "Low", "Medium",
                f"字段 {column} 几乎每行唯一，可能是标识符而非可泛化特征。",
                "人工确认字段语义，并避免无意将记录标识符直接用于模型。",
                column=column, evidence=f"唯一值比例 {unique_ratio:.2%}。",
                count=unique_count, rate=unique_ratio, sample_values=_samples(stripped),
            ))
        if row_count >= MIN_RARE_CATEGORY_ROWS and 1 < unique_count <= 100:
            rare = counts[counts / row_count < RARE_CATEGORY_THRESHOLD]
            if not rare.empty:
                issues.append(_issue(
                    "DQ_RARE_CATEGORIES", "Categorical Quality", "Rare Categories",
                    "Low", "Medium",
                    f"字段 {column} 存在占比低于 {RARE_CATEGORY_THRESHOLD:.0%} 的稀有类别。",
                    "结合业务语义评估稀有类别的合并、保留或分层抽样策略。",
                    column=column,
                    evidence=f"{len(rare)} 个稀有类别，共 {int(rare.sum())} 行。",
                    count=int(rare.sum()), rate=float(rare.sum() / row_count),
                    sample_values=_samples(rare.index),
                ))


def _duplicate_split_analysis(df, row_hashes, data_flow):
    duplicates_exist = bool(row_hashes.duplicated(keep=False).any())
    split = (data_flow or {}).get("split", {})
    base = {
        "status": "Unable to quantify",
        "duplicate_groups": int(row_hashes.value_counts().gt(1).sum()),
        "cross_split_duplicate_groups": None,
        "shared_records": None,
        "sample_rows": [],
        "reason": None,
    }
    if not duplicates_exist:
        return {**base, "status": "Exact", "cross_split_duplicate_groups": 0, "shared_records": 0}
    if not split.get("detected"):
        return {**base, "reason": "未检测到可重建的 train_test_split。"}
    if split.get("random_state") is None:
        return {**base, "reason": "random_state 未知，无法精确恢复 train/test 重叠。"}
    try:
        from sklearn.model_selection import train_test_split

        indices = list(range(len(df)))
        stratify = None
        target_column = ((data_flow or {}).get("target") or {}).get("column")
        if split.get("stratify") and target_column in df.columns:
            stratify = df[target_column]
        train_indices, test_indices = train_test_split(
            indices,
            test_size=split.get("test_size") if split.get("test_size") is not None else 0.25,
            random_state=split.get("random_state"),
            stratify=stratify,
        )
        train_counts = row_hashes.iloc[train_indices].value_counts()
        test_counts = row_hashes.iloc[test_indices].value_counts()
        shared = set(train_counts.index) & set(test_counts.index)
        shared = {value for value in shared if int((row_hashes == value).sum()) > 1}
        rows = [
            int(index) for index, value in row_hashes.items() if value in shared
        ]
        return {
            **base,
            "status": "Exact",
            "cross_split_duplicate_groups": len(shared),
            "shared_records": sum(int(train_counts[value] + test_counts[value]) for value in shared),
            "sample_rows": rows[:20],
            "reason": None,
        }
    except Exception as exc:
        return {**base, "reason": f"划分重建失败：{exc}"}


def analyze_csv(csv_path, constraints=None, business_rules=None, data_flow=None):
    """Analyze a CSV without modifying it; constraints are optional explicit business rules."""
    constraints = {**(constraints or {}), **(business_rules or {})}
    df, raw_header = _read_csv(csv_path)
    rows, columns = df.shape
    issues = []

    numeric_df = df.select_dtypes(include="number")
    result = {
        "shape": {"rows": rows, "columns": columns},
        "missing_values": {},
        "duplicate_rows": 0,
        "duplicate_rate": 0.0,
        "duplicate_groups": 0,
        "duplicate_group_details": [],
        "duplicate_split_analysis": {},
        "data_types": df.dtypes.astype(str).to_dict(),
        "correlation": {},
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": [column for column in df.columns if column not in numeric_df.columns],
        "issues": issues,
        "outliers": [],
        "invalid_values": [],
        "type_issues": [],
        "categorical_issues": [],
        "feature_issues": [],
        "missing_issues": [],
        "high_correlation_pairs": [],
        "numeric_statistics": {},
        "numeric_variable_types": {},
        "categorical_profiles": {},
        "analysis_limits": [],
    }

    if rows == 0 or columns == 0:
        issues.append(_issue(
            "DQ_EMPTY_DATASET", "Structure", "Empty Dataset", "High", "High",
            "数据集没有可分析的记录或字段。",
            "检查导出过程、分隔符与上游数据生成步骤。",
            evidence=f"检测到 {rows} 行 × {columns} 列。", count=0, rate=1.0,
        ))
        return result

    if rows < VERY_SMALL_ROW_THRESHOLD or columns >= rows:
        issues.append(_issue(
            "DQ_VERY_SMALL_DATASET", "Structure", "Very Small Dataset", "Low", "High",
            "样本量较少，或样本数相对于特征数不足。",
            "将此项视为建模稳定性提醒，并结合任务复杂度评估所需样本量。",
            evidence=f"{rows} 行 × {columns} 列。", count=rows,
            rate=columns / max(1, rows),
        ))

    header_counts = Counter(raw_header)
    for name, count in header_counts.items():
        if count > 1:
            issues.append(_issue(
                "DQ_DUPLICATE_COLUMN_NAME", "Structure", "Duplicate Column Names",
                "Medium", "High",
                f"CSV 表头中字段 {name!r} 重复出现。",
                "为重复字段指定明确且唯一的名称，避免读取时被自动重命名。",
                column=name, evidence=f"表头出现 {count} 次。",
                count=count, sample_values=[name],
            ))

    missing = df.isna().mean()
    result["missing_values"] = missing[missing > 0].to_dict()
    for column, rate in result["missing_values"].items():
        count = int(df[column].isna().sum())
        issues.append(_issue(
            "DQ_MISSING_VALUES", "Missing Values", "Missing Values",
            _severity_for_rate(float(rate), medium=0.10, high=0.50), "High",
            f"字段 {column} 存在真实缺失值。",
            "结合缺失机制、目标变量与数据生成过程选择填补、建模或保留策略。",
            column=column, evidence=f"{count}/{rows} 个值为 NaN/None（{rate:.2%}）。",
            count=count, rate=rate,
        ))

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            issues.append(_issue(
                "DQ_ALL_NULL_COLUMN", "Feature Quality", "All-null Column",
                "Medium", "High", f"字段 {column} 全部为缺失值。",
                "核对上游生成逻辑与字段必要性；不要自动删除原始数据。",
                column=column, evidence=f"{rows}/{rows} 个值缺失。", count=rows, rate=1.0,
            ))
        else:
            value_counts = non_null.value_counts(dropna=True)
            unique_count = int(non_null.nunique(dropna=True))
            if unique_count == 1:
                issues.append(_issue(
                    "DQ_CONSTANT_COLUMN", "Feature Quality", "Constant Column",
                    "Low", "High", f"字段 {column} 只有一个有效值。",
                    "确认字段是否仍具业务或审计用途；不要在未确认前自动删除。",
                    column=column, evidence=f"唯一有效值：{_clean_scalar(non_null.iloc[0])}",
                    count=int(non_null.count()), rate=1.0, sample_values=_samples(non_null),
                ))
            elif not value_counts.empty:
                dominant_rate = float(value_counts.iloc[0] / len(non_null))
                if dominant_rate >= NEAR_CONSTANT_THRESHOLD:
                    issues.append(_issue(
                        "DQ_NEAR_CONSTANT_COLUMN", "Feature Quality", "Near-constant Column",
                        "Low", "High", f"字段 {column} 的单一取值占比非常高。",
                        "评估该字段的方差、业务含义和建模贡献。",
                        column=column,
                        evidence=f"值 {_clean_scalar(value_counts.index[0])!r} 占 {dominant_rate:.2%}。",
                        count=int(value_counts.iloc[0]), rate=dominant_rate,
                        sample_values=_samples(value_counts.index),
                    ))

        normalized_name = str(column).strip().casefold()
        if normalized_name == "index" or normalized_name.startswith("unnamed:"):
            numeric = pd.to_numeric(series, errors="coerce")
            sequential = numeric.notna().all() and len(numeric) > 0 and all(
                float(value) == index for index, value in enumerate(numeric.tolist())
            )
            if sequential or series.nunique(dropna=True) / max(1, rows) >= POTENTIAL_ID_THRESHOLD:
                issues.append(_issue(
                    "DQ_POTENTIAL_INDEX_COLUMN", "Structure", "Potential Index Column",
                    "Low", "High" if sequential else "Medium",
                    f"字段 {column} 看起来像导出时保留的索引列。",
                    "确认该字段是否有业务含义；若只是导出索引，可在建模输入中排除。",
                    column=column,
                    evidence="值与 0..N-1 顺序完全一致。" if sequential else "字段名和唯一值比例均类似索引。",
                    count=rows, rate=1.0, sample_values=_samples(series),
                ))

        if not pd.api.types.is_numeric_dtype(series):
            text = series.dropna().astype(str)
            stripped = text.str.strip()
            blank = stripped == ""
            if blank.any():
                issues.append(_issue(
                    "DQ_BLANK_STRING", "Missing Values", "Blank String",
                    "Medium" if blank.mean() >= 0.10 else "Low", "High",
                    f"字段 {column} 包含空字符串或仅空白字符。",
                    "根据字段语义决定是否映射为缺失值，并保留转换记录。",
                    column=column, evidence=f"{int(blank.sum())} 个空白字符串。",
                    count=int(blank.sum()), rate=float(blank.sum() / rows),
                    sample_values=_samples(text[blank]),
                ))
            lowered = stripped.str.casefold()
            for token, count in lowered[lowered.isin(MISSING_TOKENS)].value_counts().items():
                actual = stripped[lowered == token]
                rate = int(count) / rows
                issues.append(_issue(
                    "DQ_SUSPECTED_MISSING_TOKEN", "Missing Values", "Suspected Missing Token",
                    "Low", "Medium", f"字段 {column} 中出现疑似缺失占位符 {token!r}。",
                    "先确认业务语义，再决定是否将该 token 映射为真正缺失值。",
                    column=column, evidence=f"token={token!r}，出现 {int(count)} 次（{rate:.2%}）。",
                    count=int(count), rate=rate, sample_values=_samples(actual), token=token,
                ))

    row_hashes = pd.util.hash_pandas_object(df, index=False)
    hash_counts = row_hashes.value_counts()
    duplicate_hashes = set(hash_counts[hash_counts > 1].index)
    result["duplicate_rows"] = int(df.duplicated().sum())
    result["duplicate_rate"] = result["duplicate_rows"] / rows
    result["duplicate_groups"] = len(duplicate_hashes)
    result["duplicate_group_details"] = [
        {
            "group": position + 1,
            "size": int(hash_counts[value]),
            "rows": [int(index) for index, row_hash in row_hashes.items() if row_hash == value][:10],
        }
        for position, value in enumerate(list(duplicate_hashes)[:20])
    ]
    if result["duplicate_rows"]:
        rate = result["duplicate_rate"]
        issues.append(_issue(
            "DQ_DUPLICATE_ROWS", "Duplicates", "Potential Duplicate Records",
            _severity_for_rate(rate, medium=0.01, high=0.10), "High",
            "数据集中存在内容完全相同的重复记录。",
            "核对重复记录的业务主键和生成原因；不要自动删除。",
            evidence=f"{result['duplicate_rows']}/{rows} 行完全重复（{rate:.2%}）。",
            count=result["duplicate_rows"], rate=rate,
            issue_origin="Exact row hashing", issue_nature="Needs Review", action="Review",
        ))
    result["duplicate_split_analysis"] = _duplicate_split_analysis(df, row_hashes, data_flow)
    split_duplicates = result["duplicate_split_analysis"]
    if split_duplicates.get("status") == "Exact" and split_duplicates.get("cross_split_duplicate_groups"):
        issues.append(_issue(
            "DQ_CROSS_SPLIT_DUPLICATES", "Duplicates", "Cross-split Duplicate Risk",
            "Medium", "High",
            "可重建划分中，相同记录同时出现在训练集与测试集。",
            "在划分前按业务主键或完整记录审查重复来源；经确认后再去重。",
            evidence=(
                f"重复组 {result['duplicate_groups']}；跨划分重复组 "
                f"{split_duplicates['cross_split_duplicate_groups']}；共享记录 "
                f"{split_duplicates['shared_records']}。"
            ),
            count=split_duplicates["shared_records"],
            rate=split_duplicates["shared_records"] / rows,
            affected_rows=split_duplicates.get("sample_rows", []),
            issue_origin="Reconstructed train_test_split",
            issue_nature="Needs Review",
            action="Review",
        ))

    _detect_identical_columns(df, issues)
    _detect_string_and_type_quality(df, issues, result["categorical_profiles"])

    numeric_columns = result["numeric_columns"]
    finite_numeric = pd.DataFrame(index=df.index)
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        result["numeric_statistics"][column] = _numeric_statistics(df[column])
        finite_mask = series.notna() & series.map(lambda value: math.isfinite(float(value)))
        finite = series[finite_mask]
        finite_numeric[column] = series.where(finite_mask)
        non_finite_mask = series.notna() & ~series.map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else True
        )
        if non_finite_mask.any():
            count = int(non_finite_mask.sum())
            rate = count / rows
            issues.append(_issue(
                "DQ_NON_FINITE_NUMERIC", "Invalid Values", "Non-finite Numeric Value",
                "High" if rate >= 0.05 else "Medium", "High",
                f"数值字段 {column} 包含 +inf 或 -inf。",
                "在进入统计或建模前追踪其计算来源，并显式处理非有限值。",
                column=column, evidence=f"{count} 个非有限数值（{rate:.2%}）。",
                count=count, rate=rate, sample_values=_samples(series[non_finite_mask]),
            ))

        stats = result["numeric_statistics"][column]
        variable_profile = _numeric_variable_profile(column, finite)
        result["numeric_variable_types"][column] = variable_profile
        distribution_only = variable_profile["type"] != "continuous numeric"
        skewness = stats.get("skewness")
        if (not distribution_only and skewness is not None
                and len(finite) >= MIN_OUTLIER_SAMPLES and abs(float(skewness)) >= 3):
            issues.append(_issue(
                "DQ_EXTREME_SKEWNESS", "Numeric Quality", "Extreme Skewness",
                "Low", "Medium", f"数值字段 {column} 的分布明显偏斜。",
                "结合业务分布检查变换、分层或稳健统计方法；偏斜本身不是错误。",
                column=column, evidence=f"skewness={float(skewness):.3f}", count=len(finite),
            ))

        if len(finite) >= MIN_OUTLIER_SAMPLES:
            q1, q3 = finite.quantile([0.25, 0.75])
            iqr = q3 - q1
            method = None
            lower = upper = None
            outlier_mask = pd.Series(False, index=series.index)
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = finite_mask & ((series < lower) | (series > upper))
                method = "IQR (1.5×IQR)"
            else:
                median = finite.median()
                mad = (finite - median).abs().median()
                if mad > 0:
                    robust_z = 0.6745 * (series - median).abs() / mad
                    outlier_mask = finite_mask & (robust_z > 3.5)
                    method = "MAD robust z-score"
            if method and outlier_mask.any():
                count = int(outlier_mask.sum())
                rate = count / max(1, len(finite))
                if distribution_only:
                    issues.append(_issue(
                        "DQ_DISCRETE_DISTRIBUTION_REVIEW", "Outliers",
                        "Discrete / Count-like Distribution Review", "Low", "Low",
                        f"IQR 标记了 {count} 条记录，但字段 {column} 更像 {variable_profile['type']}。",
                        "把这些值作为分布审查线索；确认业务语义前不要删除、截尾或裁剪。",
                        column=column,
                        evidence=(
                            f"IQR flags {count} records；variable_type={variable_profile['type']}；"
                            "这些值可能是合法类别或计数，而非错误记录。"
                        ),
                        count=count, rate=rate, sample_values=_samples(series[outlier_mask]),
                        affected_rows=[int(index) for index in series[outlier_mask].index[:20]],
                        issue_origin=f"Type-aware review after {method}",
                        issue_nature="Distribution Review", action="Review",
                        method=method, variable_type=variable_profile["type"],
                        lower_bound=_clean_scalar(lower), upper_bound=_clean_scalar(upper),
                    ))
                else:
                    issues.append(_issue(
                        "DQ_POTENTIAL_EXTREME_VALUES", "Outliers", "Potential Statistical Outlier",
                        "Medium" if rate >= 0.10 else "Low", "Medium",
                        f"数值字段 {column} 存在统计意义上的潜在极端值。",
                        "结合业务范围、采集过程与稳健模型判断；极端值不等于非法值。",
                        column=column,
                        evidence=(f"方法={method}；范围=[{_clean_scalar(lower)}, {_clean_scalar(upper)}]；"
                                  f"min={_clean_scalar(finite.min())}，max={_clean_scalar(finite.max())}。"),
                        count=count, rate=rate, sample_values=_samples(series[outlier_mask]),
                        affected_rows=[int(index) for index in series[outlier_mask].index[:20]],
                        issue_origin=f"Statistical method: {method}",
                        issue_nature="Statistical Anomaly", action="Review",
                        method=method, variable_type=variable_profile["type"],
                        lower_bound=_clean_scalar(lower), upper_bound=_clean_scalar(upper),
                        minimum=_clean_scalar(finite.min()), maximum=_clean_scalar(finite.max()),
                    ))

        constraint = constraints.get(column, {})
        if constraint and not finite.empty:
            violation = pd.Series(False, index=series.index)
            if constraint.get("min") is not None:
                violation |= finite_mask & (series < constraint["min"])
            if constraint.get("max") is not None:
                violation |= finite_mask & (series > constraint["max"])
            if violation.any():
                count = int(violation.sum())
                rate = count / max(1, len(finite))
                issues.append(_issue(
                    "DQ_CONSTRAINT_VIOLATION", "Invalid Values", "Confirmed Rule Violation",
                    "High" if rate >= 0.10 else "Medium", "High",
                    f"字段 {column} 有值超出显式业务约束。",
                    "修复上游数据或经业务确认后处理违规值。",
                    column=column, evidence=f"约束={constraint}；违规 {count} 个（{rate:.2%}）。",
                    count=count, rate=rate, sample_values=_samples(series[violation]),
                    affected_rows=[int(index) for index in series[violation].index[:20]],
                    issue_origin="Explicit business rule", issue_nature="Rule Violation", action="Fix",
                    constraint=dict(constraint),
                ))
        normalized_name = str(column).strip().casefold().replace(" ", "_")
        has_explicit_non_negative_rule = bool(
            constraint and constraint.get("min") is not None and constraint.get("min") >= 0
        )
        if not has_explicit_non_negative_rule and NON_NEGATIVE_NAME_RE.search(normalized_name):
            negative_mask = finite_mask & (series < 0)
            if negative_mask.any():
                count = int(negative_mask.sum())
                rate = count / max(1, len(finite))
                rows_affected = [int(index) for index in series[negative_mask].index[:20]]
                issues.append(_issue(
                    "DQ_POTENTIAL_NON_NEGATIVE_VIOLATION", "Business Rules",
                    "Potential Business Rule Violation", "Medium", "Medium",
                    f"字段 {column} 出现负值；字段语义通常暗示非负约束，但尚无显式业务规则。",
                    "请业务负责人确认 value >= 0；确认后将规则加入 business_rules。",
                    column=column,
                    evidence=(f"推断规则=value >= 0；违规 {count} 个（{rate:.2%}）；"
                              f"行号={rows_affected}。"),
                    count=count, rate=rate, sample_values=_samples(series[negative_mask]),
                    affected_rows=rows_affected,
                    issue_origin="Inferred from column semantics",
                    issue_nature="Rule Violation", action="Review",
                    rule="value >= 0",
                ))

    if len(numeric_columns) > MAX_CORRELATION_COLUMNS:
        result["analysis_limits"].append(
            f"相关性分析仅使用前 {MAX_CORRELATION_COLUMNS} 个数值字段，共检测到 {len(numeric_columns)} 个。"
        )
    correlation_columns = numeric_columns[:MAX_CORRELATION_COLUMNS]
    correlation = finite_numeric[correlation_columns].corr() if correlation_columns else pd.DataFrame()
    result["correlation"] = correlation.round(4).fillna(0).to_dict() if not correlation.empty else {}
    for index, left in enumerate(correlation.columns):
        for right in correlation.columns[index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= HIGH_CORRELATION_THRESHOLD:
                pair = {"feature_a": left, "feature_b": right, "correlation": round(float(value), 6)}
                result["high_correlation_pairs"].append(pair)
                issues.append(_issue(
                    "DQ_HIGH_CORRELATION", "Correlation", "Highly Correlated Features",
                    "Low", "High", f"字段 {left} 与 {right} 高度相关，可能包含冗余信息。",
                    "结合业务语义、模型类型和稳定性评估是否保留两者；不要自动删除。",
                    column=f"{left} / {right}", evidence=f"Pearson r={float(value):.4f}",
                    rate=abs(float(value)), **pair,
                ))

    for column in df.columns:
        normalized_name = str(column).strip().casefold()
        if normalized_name not in TARGET_NAMES:
            continue
        non_null = df[column].dropna()
        unique_count = int(non_null.nunique())
        if not non_null.empty and 1 < unique_count <= min(20, max(2, math.ceil(rows * 0.10))):
            counts = non_null.astype(str).value_counts()
            majority_rate = float(counts.iloc[0] / len(non_null))
            issues.append(_issue(
                "DQ_POTENTIAL_TARGET", "Target Quality", "Potential Target Column",
                "Low", "Medium",
                f"系统根据字段名和类别数量推测 {column} 可能是目标列，请人工确认。",
                "确认目标定义、标签生成时间与特征可用边界。",
                column=column, evidence=f"字段名={column!r}，唯一值 {unique_count} 个。",
                count=unique_count, rate=unique_count / rows, sample_values=_samples(counts.index),
            ))
            if majority_rate >= 0.90:
                issues.append(_issue(
                    "DQ_POTENTIAL_CLASS_IMBALANCE", "Target Quality", "Potential Class Imbalance",
                    "Medium", "Medium",
                    f"疑似目标列 {column} 的最大类别占比很高，请人工确认。",
                    "采用分层划分和适合不平衡任务的指标，并核对少数类样本质量。",
                    column=column,
                    evidence=f"最大类别 {_clean_scalar(counts.index[0])!r} 占 {majority_rate:.2%}。",
                    count=int(counts.iloc[0]), rate=majority_rate, sample_values=_samples(counts.index),
                ))

    result["outliers"] = [item for item in issues if item["category"] == "Outliers"]
    result["invalid_values"] = [item for item in issues if item["category"] == "Invalid Values"]
    result["business_rule_results"] = [item for item in issues if item["category"] == "Business Rules" or item["rule_id"] == "DQ_CONSTRAINT_VIOLATION"]
    result["type_issues"] = [item for item in issues if item["category"] in {"Type Consistency", "String Quality"}]
    result["categorical_issues"] = [item for item in issues if item["category"] in {"Categorical Quality", "Target Quality"}]
    result["feature_issues"] = [item for item in issues if item["category"] in {"Feature Quality", "Numeric Quality"}]
    result["missing_issues"] = [item for item in issues if item["category"] == "Missing Values"]
    return result


def analyze_dataset_consistency(data_results):
    """Return conservative project-level schema findings for multiple CSV results."""
    entries = list(data_results or [])
    if len(entries) < 2:
        return []
    issues = []
    reference = entries[0]
    reference_name = reference.get("file", "CSV 1")
    reference_result = reference.get("result", reference)
    reference_types = reference_result.get("data_types", {})
    reference_columns = set(reference_types)
    reference_normalized = {str(column).strip().casefold(): column for column in reference_columns}

    for entry in entries[1:]:
        name = entry.get("file", "CSV")
        result = entry.get("result", entry)
        types = result.get("data_types", {})
        columns = set(types)
        normalized = {str(column).strip().casefold(): column for column in columns}
        missing = sorted(reference_columns - columns)
        extra = sorted(columns - reference_columns)
        if missing or extra:
            issues.append(_issue(
                "DQ_SCHEMA_DIFFERENCE", "Multi-file Consistency", "Schema Difference",
                "Low", "High", f"{reference_name} 与 {name} 的字段集合不同。",
                "确认差异是否来自 train/test 目标列、版本变化或导出遗漏。",
                evidence=f"{name} 缺少 {missing[:10]}；新增 {extra[:10]}。",
                count=len(missing) + len(extra), sample_values=(missing + extra)[:5],
                files=[reference_name, name],
            ))
        for normalized_name in sorted(set(reference_normalized) & set(normalized)):
            left = reference_normalized[normalized_name]
            right = normalized[normalized_name]
            if left != right:
                issues.append(_issue(
                    "DQ_COLUMN_NAME_FORMAT", "Multi-file Consistency",
                    "Column Name Formatting Difference", "Low", "High",
                    "两个文件中的同名字段存在大小写或首尾空格差异。",
                    "统一字段命名规范，避免 join/concat 时产生意外列。",
                    column=f"{left} / {right}", evidence=f"{reference_name}: {left!r}；{name}: {right!r}",
                    count=1, files=[reference_name, name],
                ))
            if reference_types.get(left) != types.get(right):
                issues.append(_issue(
                    "DQ_CROSS_FILE_DTYPE", "Multi-file Consistency", "Cross-file Dtype Difference",
                    "Medium", "High", f"同名字段 {left} 在不同文件中的 dtype 不一致。",
                    "检查解析失败、缺失占位符和导出格式，并显式统一类型。",
                    column=left,
                    evidence=f"{reference_name}: {reference_types.get(left)}；{name}: {types.get(right)}",
                    count=1, files=[reference_name, name],
                ))
            left_profile = reference_result.get("categorical_profiles", {}).get(left, {})
            right_profile = result.get("categorical_profiles", {}).get(right, {})
            left_boolean = set(left_profile.get("boolean_tokens", []))
            right_boolean = set(right_profile.get("boolean_tokens", []))
            if left_boolean and right_boolean and left_boolean != right_boolean:
                left_numeric = bool(left_boolean & {"0", "1"})
                right_numeric = bool(right_boolean & {"0", "1"})
                if left_numeric != right_numeric:
                    issues.append(_issue(
                        "DQ_CROSS_FILE_CATEGORY_ENCODING", "Multi-file Consistency",
                        "Category Encoding Difference", "Low", "Medium",
                        f"字段 {left} 在文件间使用了不同的布尔类别编码。",
                        "在合并或共同建模前定义统一、可追踪的类别映射。",
                        column=left,
                        evidence=f"{reference_name}: {sorted(left_boolean)}；{name}: {sorted(right_boolean)}",
                        count=1, files=[reference_name, name],
                    ))
    return issues


def print_report(csv_path):
    report = analyze_csv(csv_path)
    print("=" * 40)
    print("数据质量报告")
    print("=" * 40)
    print("\n数据规模:", report["shape"])
    print("\n问题数量:", len(report["issues"]))
    for issue in report["issues"]:
        print(f"- {issue['severity']} · {issue['type']} · {issue['evidence']}")


if __name__ == "__main__":
    print_report("demo_data/train.csv")
