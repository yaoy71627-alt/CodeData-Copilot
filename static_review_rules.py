"""Conservative project-context rules. Never execute uploaded code or infer host versions."""

import ast
import re
try:
    import tomllib
except ImportError:  # Python 3.10 deployments
    import tomli as tomllib
from pathlib import PurePosixPath

from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version


RULESET_VERSION = "2026-10-19-batch-b"
MANIFEST_NAMES = {"requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml"}


def has_imported_cv_call(code, aliases, cv_names):
    """Resolve cross-cell import aliases without counting comments or unused imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    bindings = dict(aliases)
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                bindings[item.asname or item.name] = f"{node.module}.{item.name}"
        elif isinstance(node, ast.Import):
            for item in node.names:
                bindings[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.Call):
            name = _name(node.func)
            head, dot, tail = name.partition('.')
            api = bindings.get(head, head) + (dot + tail if dot else '')
            if api.startswith("sklearn.model_selection.") and api.split('.')[-1] in cv_names:
                return True
    return False


def is_dependency_manifest(name):
    base = PurePosixPath(str(name)).name.lower()
    return base in MANIFEST_NAMES or bool(re.fullmatch(r"requirements[-.][\w.-]+\.txt", base))


def read_dependencies(documents):
    """Retain declarations separately: conflicting/optional environments never prove failure."""
    dependencies = {}
    for name, text in documents:
        lines = text.splitlines()
        if name.lower().endswith(".toml"):
            try:
                # Optional groups are alternative environments, not mandatory constraints.
                lines = tomllib.loads(text).get("project", {}).get("dependencies", [])
            except (ValueError, TypeError):
                continue
        for line in lines:
            value = re.sub(r"\s+#.*$", "", line).strip().removeprefix("- ")
            # Conda's single '=' pin denotes a version; channels/build strings are ignored.
            value = re.sub(r"^(pandas|scikit-learn)=(\d[^=]*?)(?:=[^=]+)?$", r"\1==\2", value)
            try:
                req = Requirement(value)
            except InvalidRequirement:
                continue
            package = req.name.lower().replace("_", "-")
            if package not in {"pandas", "scikit-learn"}:
                continue
            dependencies.setdefault(package, []).append({
                "specifier": str(req.specifier), "source": name,
                "conditional": bool(req.marker or req.extras),
            })
    return dependencies


def compatibility(package, removed, dependencies):
    declarations = (dependencies or {}).get(package, [])
    boundary = Version(removed)
    verdicts = []
    for declaration in declarations:
        specs = list(Requirement(package + declaration["specifier"]).specifier)
        if declaration.get("conditional") or not specs:
            verdicts.append("unknown")
            continue
        # Prove a bound only from explicit supported operators. Mixed ranges stay risks.
        lower = upper = False
        for spec in specs:
            try:
                version = Version(spec.version)
            except ValueError:
                continue
            if spec.operator in {">=", ">", "==", "==="} and version >= boundary:
                lower = True
            if ((spec.operator == "<" and version <= boundary)
                    or (spec.operator in {"<=", "==", "==="} and version < boundary)):
                upper = True
        verdicts.append("incompatible" if lower and not upper else "compatible" if upper and not lower else "unknown")
    state = verdicts[0] if verdicts and len(set(verdicts)) == 1 else "unknown"
    evidence = "; ".join(f"{d['source']}: {package}{d['specifier']}" for d in declarations)
    return state, evidence or "项目未提供可判定的依赖版本"


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return ""


def _finding(rule, title, node, code, message, recommendation, *, status="needs_review", severity="Medium", **extra):
    return {
        "rule_id": rule, "category": "Compatibility" if "API" in rule else "Correctness",
        "type": title, "severity": severity, "confidence": "High" if status == "confirmed_error" else "Medium",
        "line": node.lineno, "end_line": node.end_lineno,
        "source_snippet": ast.get_source_segment(code, node), "evidence": message,
        "message": message, "recommendation": recommendation,
        "verification_status": status, "action": "Fix" if status == "confirmed_error" else "Review",
        **extra,
    }


def _api_finding(rule, title, node, code, package, removed, dependencies):
    state, evidence = compatibility(package, removed, dependencies)
    # Old supported versions still deserve a migration notice, not a confirmed error.
    status = "confirmed_error" if state == "incompatible" else "needs_review"
    message = f"{title}；该接口在 {package} {removed} 起已移除。{evidence}。"
    message += "声明的版本约束与此接口不兼容（未执行用户代码）。" if status == "confirmed_error" else "尚不能认定当前运行必然失败，请核对实际环境。"
    return _finding(rule, title, node, code, message,
                    "核对项目锁定版本，迁移到受支持 API，并验证参数语义和下游行为。",
                    status=status, severity="High" if status == "confirmed_error" else "Low",
                    dependency_status=state, dependency_evidence=evidence)


def analyze_additional_rules(code, dependencies=None, initial_aliases=None, initial_frames=None):
    """Lexical AST checks; caller supplies earlier notebook cells' safe import/type context."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    aliases = dict(initial_aliases or {})
    frames = set(initial_frames or ())
    findings = []

    def resolved(node):
        value = _name(node)
        head, dot, tail = value.partition(".")
        return aliases.get(head, head) + (dot + tail if dot else "")

    # Walk statement lists in lexical order, with isolated local scopes.
    def scan(body, inherited_aliases, inherited_frames, initially_bound=()):
        nonlocal aliases, frames
        old_aliases, old_frames = aliases, frames
        aliases, frames = dict(inherited_aliases), set(inherited_frames)
        for index, node in enumerate(body):
            if isinstance(node, ast.Try):
                prior = body[:index]
                bound_before = set(initially_bound) | {n.id for stmt in prior for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                definitions = {target.id for stmt in node.body if isinstance(stmt, (ast.Assign, ast.AnnAssign))
                               for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]) if isinstance(target, ast.Name)}
                later_reads = {n.id for stmt in body[index + 1:] for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                unresolved = []
                for variable in sorted(definitions & later_reads - bound_before):
                    unsafe = False
                    for handler in node.handlers:
                        stops = any(isinstance(stmt, (ast.Raise, ast.Return)) for stmt in handler.body)
                        fallback = any(isinstance(stmt, (ast.Assign, ast.AnnAssign)) and any(isinstance(t, ast.Name) and t.id == variable for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])) for stmt in handler.body)
                        unsafe |= not stops and not fallback
                    finally_fallback = any(isinstance(stmt, (ast.Assign, ast.AnnAssign)) and any(isinstance(t, ast.Name) and t.id == variable for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])) for stmt in node.finalbody)
                    # A later assignment before the first read also establishes a fallback.
                    assigned_later = False
                    for stmt in body[index + 1:]:
                        if any(isinstance(n, ast.Name) and n.id == variable and isinstance(n.ctx, ast.Load) for n in ast.walk(stmt)):
                            break
                        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and any(isinstance(t, ast.Name) and t.id == variable for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])):
                            assigned_later = True
                    if unsafe and not finally_fallback and not assigned_later:
                        unresolved.append(variable)
                if unresolved:
                    findings.append(_finding("CODE_EXCEPTION_UNBOUND_VARIABLE", "异常后继续执行可能使用未赋值变量", node, code,
                        f"{', '.join(unresolved)} 在 try 内赋值，异常处理路径可能继续执行，后续仍读取这些变量；需复核具体异常发生点与控制流。",
                        "异常时终止当前路径，或提供明确且有效的回退值；不要仅捕获并打印异常。", status="static_risk"))
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
                    if item.name == "sklearn.cross_validation" or item.name.startswith("sklearn.cross_validation."):
                        findings.append(_api_finding("CODE_SKLEARN_LEGACY_API", "旧 sklearn.cross_validation 模块", node, code, "scikit-learn", "0.20", dependencies))
            elif isinstance(node, ast.ImportFrom):
                for item in node.names:
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
                if node.module == "sklearn.cross_validation" or (node.module == "sklearn" and any(i.name == "cross_validation" for i in node.names)):
                    findings.append(_api_finding("CODE_SKLEARN_LEGACY_API", "旧 sklearn.cross_validation 模块", node, code, "scikit-learn", "0.20", dependencies))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scoped = dict(aliases)
                scoped_frames = set(frames)
                args = getattr(node, "args", None)
                parameters = [*getattr(args, "posonlyargs", []), *getattr(args, "args", []), *getattr(args, "kwonlyargs", [])]
                parameters += [a for a in (getattr(args, "vararg", None), getattr(args, "kwarg", None)) if a]
                for arg in parameters:
                    scoped.pop(arg.arg, None)
                    scoped_frames.discard(arg.arg)
                scan(node.body, scoped, scoped_frames, {a.arg for a in parameters})
                aliases.pop(node.name, None)
                continue
            # Do not inspect nested bodies twice; recursively handle each statement list.
            roots = [value for field, value in ast.iter_fields(node) if field not in {"body", "orelse", "finalbody", "handlers"}]
            nodes = [child for root in roots for value in (root if isinstance(root, list) else [root]) if isinstance(value, ast.AST) for child in ast.walk(value)]
            for call in (n for n in nodes if isinstance(n, ast.Call)):
                api = resolved(call.func)
                if api == "sklearn.preprocessing.OneHotEncoder" and any(k.arg == "sparse" for k in call.keywords):
                    findings.append(_api_finding("CODE_ONEHOT_SPARSE_API", "OneHotEncoder 使用已移除的 sparse 参数", call, code, "scikit-learn", "1.4", dependencies))
                if isinstance(call.func, ast.Attribute) and call.func.attr == "append" and _name(call.func.value) in frames:
                    findings.append(_api_finding("CODE_PANDAS_APPEND", "DataFrame.append 接口", call, code, "pandas", "2.0", dependencies))
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Compare):
                if any(isinstance(n, ast.Attribute) and n.attr == "status_code" for n in ast.walk(node.value)):
                    findings.append(_finding("CODE_UNUSED_STATUS_COMPARISON", "状态码比较结果被丢弃", node, code,
                        "状态码比较是独立表达式；结果未被 if/assert/返回值使用，不能据此阻止失败响应继续处理。",
                        "显式检查响应状态并在失败时终止当前处理路径，或调用 raise_for_status()。", status="confirmed_error"))
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and index:
                previous = body[index - 1]
                call = node.value
                if (isinstance(previous, ast.Expr) and isinstance(previous.value, ast.Call)
                        and ast.dump(call, include_attributes=False) == ast.dump(previous.value, include_attributes=False)
                        and re.match(r"^(?:test|check|validate)_", _name(call.func))):
                    args = [a.id for a in call.args if isinstance(a, ast.Name)]
                    scope_names = {n.id for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
                    missing = [a.replace("test", "train") for a in args if "test" in a and a.replace("test", "train") in scope_names]
                    checked = {_name(c.func) + ":" + a.id for stmt in body for c in ast.walk(stmt) if isinstance(c, ast.Call) for a in c.args if isinstance(a, ast.Name)}
                    missing = [a for a in missing if _name(call.func) + ":" + a not in checked]
                    if missing:
                        findings.append(_finding("CODE_DUPLICATE_CHECK_CALL", "连续重复检查，可能遗漏对应训练变量", node, code,
                            f"同一作用域连续调用相同检查，未见同函数检查 {', '.join(missing)}；仅为待复核线索，不能认定检查必需或业务逻辑错误。",
                            "确认重复调用是否有意，并核对训练/测试输入检查的覆盖范围。", confidence="Low"))
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                is_frame = (isinstance(value, ast.Call) and (resolved(value.func) in {"pandas.DataFrame", "pandas.read_csv", "pandas.read_parquet"}
                            or (isinstance(value.func, ast.Attribute) and _name(value.func.value) in frames and value.func.attr in {"copy", "drop", "rename", "reset_index"}))) or (isinstance(value, ast.Name) and value.id in frames)
                for target in targets:
                    if isinstance(target, ast.Name):
                        if is_frame:
                            frames.add(target.id)
                        else:
                            frames.discard(target.id)
                        aliases.pop(target.id, None)
            for field in ("body", "orelse", "finalbody"):
                child_body = getattr(node, field, None)
                if isinstance(child_body, list):
                    scan(child_body, aliases, frames)
            for handler in getattr(node, "handlers", []):
                scan(handler.body, aliases, frames)
        state = (dict(aliases), set(frames))
        aliases, frames = old_aliases, old_frames
        return state

    scan(tree.body, aliases, frames)
    return findings


def notebook_context(cells):
    """Only earlier unconditional top-level imports/frame constructors, reset on reassignment."""
    aliases, frames = {}, set()
    for code in cells:
        yield dict(aliases), set(frames)
        try:
            body = ast.parse(code).body
        except SyntaxError:
            continue
        for node in body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                for item in node.names:
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name):
                        aliases.pop(target.id, None)
                        value = node.value
                        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr in {"DataFrame", "read_csv"} and aliases.get(_name(value.func.value)) == "pandas":
                            frames.add(target.id)
                        else:
                            frames.discard(target.id)


def documentation_split_findings(code_units, documents):
    """Compare explicit train/test claims only when there is one unambiguous split."""
    splits = []
    for unit in code_units:
        try:
            tree = ast.parse(unit["code"])
        except SyntaxError:
            continue
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and _name(call.func).split('.')[-1] == "train_test_split":
                value = next((k.value for k in call.keywords if k.arg == "test_size"), None)
                ratio = value.value if isinstance(value, ast.Constant) and isinstance(value.value, float) and 0 < value.value < 1 else None
                splits.append((unit, call, ratio))
    results = []
    for document in documents:
        doc_path = PurePosixPath(document["file"])
        if not doc_path.name.lower().startswith("readme"):
            continue
        scoped_splits = [s for s in splits if doc_path.parent == PurePosixPath('.') or PurePosixPath(s[0]["file"]).is_relative_to(doc_path.parent)]
        for line_number, line in enumerate(document["excerpt"].splitlines(), 1):
            match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:train|训练)\s*[/,，]\s*(\d+(?:\.\d+)?)\s*%\s*(?:test|测试)", line, re.I)
            if not match or abs(float(match[1]) + float(match[2]) - 100) > 0.01:
                continue
            candidates = [s for s in scoped_splits if s[0]["file"] in line] or scoped_splits
            if len(candidates) != 1:
                continue
            unit, call, ratio = candidates[0]
            if ratio is None:
                continue
            if abs(float(match[2]) / 100 - ratio) < 1e-8:
                continue
            issue = _finding("CODE_DOCUMENTED_SPLIT_MISMATCH", "文档与代码的训练/测试划分不一致", call, unit["code"],
                f"{document['file']}:{line_number} 声明 {match[1]}% train/{match[2]}% test；{unit['file']}:{call.lineno} 显式 test_size={ratio}。文档可能描述计划或旧版本，需确认当前流程。",
                "确认实际使用的划分入口，统一文档和代码，并记录随机种子。",
                affected_locations=[{"file": document["file"], "line": line_number, "evidence": line.strip()}])
            results.append({"file": unit["file"], **({"cell": unit["cell"]} if unit.get("cell") else {}), "issues": [issue]})
    return results


def slice_sample_findings(code_units, csv_rows):
    """Propagate row counts through a small allowlist; missing lineage remains unknown.

    csv_rows maps imported project paths to verified row counts, never model estimates.
    State is isolated by file, but persists across ordered notebook cells.
    """
    states, results, import_states = {}, [], {}
    for unit in code_units:
        frames, integers = states.setdefault(unit["file"], ({}, {}))
        aliases = import_states.setdefault(unit["file"], {})
        code = unit["code"]
        try:
            tree = ast.parse(code)
        except SyntaxError:
            frames.clear()
            continue

        def integer(node):
            if node is None:
                return None
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.Name):
                return integers.get(node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                n = integer(node.operand)
                return -n if n is not None else None
            return None

        def cardinality(value):
            if isinstance(value, ast.Name):
                return frames.get(value.id)
            if isinstance(value, ast.Subscript):
                parent = value.value
                if isinstance(parent, ast.Attribute) and parent.attr == "iloc":
                    parent = parent.value
                n = cardinality(parent)
                if isinstance(value.slice, ast.Slice):
                    bounds = [integer(value.slice.lower), integer(value.slice.upper), integer(value.slice.step)]
                    if n is not None and all(node is None or bound is not None for node, bound in zip((value.slice.lower, value.slice.upper, value.slice.step), bounds)) and bounds[2] != 0:
                        return len(range(*slice(*bounds).indices(n)))
                # Literal column selection is row-preserving; Boolean masks are not.
                elif isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str):
                    return n
                elif isinstance(value.slice, (ast.List, ast.Tuple)) and all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in value.slice.elts):
                    return n
                return None
            if isinstance(value, ast.Call):
                name = _name(value.func)
                head, dot, tail = name.partition('.')
                api = aliases.get(head, head) + (dot + tail if dot else '')
                if api == "pandas.read_csv" and value.args and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str) and not value.keywords:
                    name = PurePosixPath(value.args[0].value.replace('\\', '/')).name
                    matches = [rows for path, rows in csv_rows.items() if PurePosixPath(path).name == name]
                    return matches[0] if len(matches) == 1 else None
                if isinstance(value.func, ast.Attribute):
                    n = cardinality(value.func.value)
                    if value.func.attr in {"copy", "rename", "reset_index"}:
                        return n
                    if value.func.attr == "drop" and any(k.arg == "columns" or (k.arg == "axis" and isinstance(k.value, ast.Constant) and k.value.value in (1, "columns")) for k in value.keywords):
                        return n
            return None

        def row_mask(value):
            if isinstance(value, ast.Compare):
                return any(isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.value, ast.Name) and n.value.id in frames for n in ast.walk(value))
            if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.BitAnd, ast.BitOr)):
                return row_mask(value.left) and row_mask(value.right)
            return False

        for node in tree.body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                for item in node.names:
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
            expression = node.value if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) else None
            if expression is not None:
                for call in (c for c in ast.walk(expression) if isinstance(c, ast.Call)):
                    name = _name(call.func)
                    head, dot, tail = name.partition('.')
                    api = aliases.get(head, head) + (dot + tail if dot else '')
                    passed_frames = any(isinstance(arg, ast.Name) and arg.id in frames for arg in [*call.args, *(k.value for k in call.keywords)])
                    receiver = _name(call.func.value) if isinstance(call.func, ast.Attribute) else None
                    safe_external = api in {"print", "len", "display", "sklearn.feature_selection.mutual_info_classif", "sklearn.feature_selection.chi2"} or api.startswith(("seaborn.", "matplotlib.pyplot."))
                    safe_method = isinstance(call.func, ast.Attribute) and call.func.attr in {"copy", "head", "tail", "describe", "info", "corr", "isnull", "isna", "sum", "value_counts", "drop", "dropna", "rename", "reset_index", "groupby", "select_dtypes", "to_csv"}
                    if (passed_frames and not safe_external) or (receiver in frames and not safe_method):
                        frames.clear()
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                n = cardinality(node.value)
                for target in targets:
                    if not isinstance(target, ast.Name):
                        root = target
                        while isinstance(root, (ast.Subscript, ast.Attribute)):
                            root = root.value
                        if not isinstance(root, ast.Name) or root.id not in frames:
                            continue
                        # Known column writes cannot append rows; other index writes can.
                        safe_column = isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)
                        safe_mask_column = (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                                            and target.value.attr == "loc" and isinstance(target.slice, ast.Tuple)
                                            and len(target.slice.elts) == 2 and row_mask(target.slice.elts[0])
                                            and isinstance(target.slice.elts[1], ast.Constant) and isinstance(target.slice.elts[1].value, str))
                        if not safe_column and not safe_mask_column:
                            frames.clear()
                        continue
                    # Snapshot parent cardinality before rebinding the target.
                    value = node.value
                    if "test" in target.id.casefold() and isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
                        lower, upper = integer(value.slice.lower), integer(value.slice.upper)
                        if (n is not None and n < 20) or (n is None and lower is not None and value.slice.upper is None):
                            size = f"已导入数据和允许列表内的行数变换可验证该切片为 {n} 行。" if n is not None else "未能关联并验证切片来源行数，不能据起点推断剩余样本量。"
                            issue = _finding("CODE_SLICE_TEST_SAMPLE", "切片测试样本规模需要复核", node, code,
                                f"{target.id} 使用位置切片。{size}若用于泛化评价，样本量可能不足；若仅为部署连通性烟测，不能据此认定评价流程错误。",
                                "确认该切片用于部署烟测还是独立评价；评价集应具代表性，并报告不确定性。",
                                severity="High" if n is not None and n <= 1 else "Medium",
                                **({"verified_test_rows": n} if n is not None else {}))
                            results.append({"file": unit["file"], **({"cell": unit["cell"]} if unit.get("cell") else {}), "issues": [issue]})
                    frames.pop(target.id, None)
                    integers.pop(target.id, None)
                    aliases.pop(target.id, None)
                    if n is not None:
                        frames[target.id] = n
                    number = integer(node.value)
                    if number is not None:
                        integers[target.id] = number
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute):
                    receiver = _name(call.func.value)
                    if receiver in frames and any(k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords):
                        if cardinality(call) is None:
                            frames.clear()  # Aliases may share an in-place-mutated object.
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                # Branches/loops are not replayed. Writes may change row cardinality.
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        frames.pop(child.id, None)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and any(k.arg == "inplace" for k in child.keywords):
                        frames.pop(_name(child.func.value), None)
            elif isinstance(node, ast.Delete):
                frames.clear()
    return results
