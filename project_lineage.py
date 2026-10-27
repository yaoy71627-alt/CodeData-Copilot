"""Project CSV read/write associations from literal AST evidence, never execution."""
import ast
from copy import deepcopy
import posixpath
from pathlib import PurePosixPath
import re
import shlex

from data_flow_analyzer import analyze_python_data_flow


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + '.' + node.attr
    return ''


def _path(value):
    if not isinstance(value, str) or not value or '\x00' in value:
        return None
    # Logical archive prefixes remain part of identity and cannot escape to siblings.
    return posixpath.normpath(value.replace('\\', '/'))


def resolve_csv(value, code_file, names):
    value = _path(value)
    if not value:
        return None, 'dynamic_path'
    namespace = code_file.rsplit('!/', 1)[0] + '!/' if '!/' in code_file else ''
    local_file = code_file.rsplit('!/', 1)[-1]
    root_value = _path(namespace + value)
    relative_value = _path(namespace + posixpath.join(posixpath.dirname(local_file), value))
    exact = sorted({n for n in names if n in {root_value, relative_value} and (n.startswith(namespace) if namespace else '!/' not in n)})
    if len(exact) == 1:
        return exact[0], 'explicit_path'
    if len(exact) > 1:
        return None, 'ambiguous_path'
    if '/' in value and not value.startswith('/') and not re.match(r'^[A-Za-z]:/', value) and '://' not in value:
        suffix_matches = [n for n in names if n.endswith('/' + value) and (n.startswith(namespace) if namespace else '!/' not in n)]
        return (suffix_matches[0], 'project_relative_suffix') if len(suffix_matches) == 1 else (None, 'ambiguous_path' if suffix_matches else 'not_imported')
    basename = PurePosixPath(value).name
    matches = [n for n in names if PurePosixPath(n).name == basename and (n.startswith(namespace) if namespace else '!/' not in n)]
    return (matches[0], 'unique_basename') if len(matches) == 1 else (None, 'ambiguous_path' if matches else 'not_imported')


def _literal(node, values):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.BinOp):
        left, right = _literal(node.left, values), _literal(node.right, values)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Div):
                return posixpath.join(left, right)
    if isinstance(node, ast.Call):
        name = _name(node.func)
        arguments = [_literal(a, values) for a in node.args]
        if name in {'Path', 'pathlib.Path', 'str'} and len(arguments) == 1:
            return arguments[0]
        if name in {'os.path.join', 'posixpath.join'} and arguments and all(v is not None for v in arguments):
            return posixpath.join(*arguments)
    return None


def _cli_bindings(documents):
    """Only explicit Python commands with literal flags; no shell expansion or execution."""
    bindings = []
    for doc in documents:
        text = doc['excerpt'].replace('\\\n', ' ')
        for lineno, line in enumerate(text.splitlines(), 1):
            try:
                args = shlex.split(line.strip())
            except ValueError:
                continue
            if not args or not re.fullmatch(r'python(?:3(?:\.\d+)?)?', args[0]) or len(args) < 2 or not args[1].endswith('.py'):
                continue
            options = {}
            i = 2
            while i < len(args):
                token = args[i]
                if token.startswith('--'):
                    key, equal, value = token.partition('=')
                    if not equal and i + 1 < len(args):
                        i += 1
                        value = args[i]
                    if value and not any(c in value for c in '$<>`|;&'):
                        options[key] = value
                i += 1
            bindings.append({'script': _path(posixpath.join(posixpath.dirname(doc['file']), args[1])), 'options': options,
                             'evidence': {'file': doc['file'], 'line': lineno, 'command_type': 'literal_python_call'}})
    return bindings


def build_dataset_graph(code_units, csv_files, documents=()):
    names = sorted(name for name, _ in csv_files)
    reads, writes, transformations, consumers = [], [], [], []
    bindings = _cli_bindings(documents)
    states = {}
    for unit in code_units:
        try:
            tree = ast.parse(unit['code'])
        except SyntaxError:
            continue
        file = unit['file']
        state = states.setdefault(file, {'values': {}, 'origins': {}, 'roles': {}, 'aliases': {}})
        aliases = state['aliases']
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                for item in node.names:
                    aliases[item.asname or item.name] = f'{node.module}.{item.name}'

        def resolved(node):
            head, dot, tail = _name(node).partition('.')
            return aliases.get(head, head) + (dot + tail if dot else '')

        def visit(body, values, origins, roles, context=None):
            def location(node):
                return {'file': file, 'cell': unit.get('cell'), 'line': node.lineno,
                        **({'binding': context} if context else {})}

            def sources(expr):
                return sorted({o for n in ast.walk(expr) if isinstance(n, ast.Name) for o in origins.get(n.id, [])})

            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = [a.arg for a in node.args.args]
                    invocations = []
                    # Literal callsites in this file, including docopt option bindings.
                    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _name(n.func) == node.name]
                    cli = [b for b in bindings if b['script'] == file]
                    for call in calls:
                        if len(call.args) > len(params):
                            continue
                        for command in cli or [None]:
                            env = dict(values)
                            for param, arg in zip(params, call.args):
                                value = _literal(arg, values)
                                if command and isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant):
                                    value = command['options'].get(arg.slice.value)
                                if value is not None:
                                    env[param] = value
                            for kw in call.keywords:
                                value = _literal(kw.value, values)
                                if kw.arg in params and value is not None:
                                    env[kw.arg] = value
                            invocations.append((env, command['evidence'] if command else None))
                    saved_aliases = dict(aliases)
                    for param in params:
                        aliases.pop(param, None)
                    for env, evidence in invocations or [(dict(values), None)]:
                        visit(node.body, env, {}, {}, evidence)
                    aliases.clear()
                    aliases.update(saved_aliases)
                    continue
                expression = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
                if expression is not None:
                    for call in (c for c in ast.walk(expression) if isinstance(c, ast.Call)):
                        if isinstance(call.func, ast.Attribute) and call.func.attr in {'fit', 'predict', 'fit_transform', 'transform'}:
                            consumers.append({**location(node), 'call': _name(call.func),
                                              'variables': [_name(arg) for arg in call.args],
                                              'sources': sorted({s for arg in call.args for s in sources(arg)}),
                                              'verification': 'static_dependency_only'})
                if isinstance(node, ast.Assign):
                    value = node.value
                    targets = [n.id for t in node.targets for n in ([t] if isinstance(t, ast.Name) else t.elts if isinstance(t, (ast.Tuple, ast.List)) else []) if isinstance(n, ast.Name)]
                    # Expression result is a transformation trace, not proof of row equivalence.
                    parent_sources = sources(value)
                    split = isinstance(value, ast.Call) and resolved(value.func) == 'sklearn.model_selection.train_test_split'
                    for i, target in enumerate(targets):
                        literal = _literal(value, values)
                        values.pop(target, None)
                        if literal is not None:
                            values[target] = literal
                        origins[target] = parent_sources
                        roles.pop(target, None)
                        if split:
                            roles[target] = 'train' if i % 2 == 0 else 'test'
                        if isinstance(value, ast.Call) and resolved(value.func) in {'pandas.read_csv', 'pandas.read_table'}:
                            expression = value.args[0] if value.args else next((k.value for k in value.keywords if k.arg in {'filepath_or_buffer', 'path'}), None)
                            path = _literal(expression, values)
                            dataset, method = resolve_csv(path, file, names)
                            hint = _name(expression).lower() + ' ' + target.lower()
                            role = 'train' if 'train' in hint else 'test' if 'test' in hint else 'unknown'
                            origins[target] = [dataset] if dataset else []
                            roles[target] = role
                            reads.append({**location(node), 'variable': target, 'path': path,
                                          'dataset': dataset, 'match': method, 'role': role,
                                          'replay_safe': resolved(value.func) == 'pandas.read_csv' and len(value.args) <= 1
                                          and all(k.arg == 'filepath_or_buffer' for k in value.keywords)})
                        if parent_sources and not (isinstance(value, ast.Call) and resolved(value.func) == 'pandas.read_csv'):
                            transformations.append({**location(node), 'output': target,
                                                    'sources': parent_sources, 'operation': 'split' if split else 'transformation'})
                        aliases.pop(target, None)
                    continue
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == 'to_csv':
                        variable = _name(call.func.value)
                        expression = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == 'path_or_buf'), None)
                        path = _literal(expression, values)
                        dataset, method = resolve_csv(path, file, names)
                        writes.append({**location(node), 'variable': variable, 'path': path,
                                       'dataset': dataset, 'match': method, 'role': roles.get(variable, 'derived'),
                                       'sources': list(origins.get(variable, []))})
                for key in ('body', 'orelse', 'finalbody'):
                    child = getattr(node, key, None)
                    if child:
                        visit(child, dict(values), origins, roles, context)
                for handler in getattr(node, 'handlers', []):
                    visit(handler.body, dict(values), origins, roles, context)

        visit(tree.body, state['values'], state['origins'], state['roles'])
    # Remove identical events (e.g. a repeated callsite with the same resolved flags).
    import json
    def unique(records):
        return list({json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in records}.values())
    reads, writes = unique(reads), unique(writes)
    datasets = []
    for name in names:
        loaders = [r for r in reads if r['dataset'] == name]
        producers = [r for r in writes if r['dataset'] == name]
        parents = sorted({s for w in producers for s in w['sources'] if s != name})
        roles = sorted(({r['role'] for r in loaders} | {r['role'] for r in producers if parents or r['role'] in {'train', 'test'}}) - {'unknown'})
        datasets.append({'file': name, 'loaders': loaders, 'producers': producers, 'parents': parents,
                         'consumers': [c for c in consumers if name in c['sources']],
                         'roles': roles or ['raw' if loaders else 'unassociated'],
                         'association_status': 'matched' if loaders or producers else 'unassociated'})
    return {'datasets': datasets, 'reads': reads, 'writes': writes, 'transformations': unique(transformations), 'consumers': unique(consumers),
            'unresolved_reads': [r for r in reads if not r['dataset']],
            'statement': '仅按静态路径和明确命令关联；读写依赖不证明文件内容与某次运行完全一致。'}


def bind_dataset_flow(graph, name, csv_path, flows):
    """Choose only an unambiguous single-source replay context, never highest score."""
    candidates = []
    for flow in flows:
        reads = [r for r in graph['reads'] if r['file'] == flow.get('file')]
        if not reads or any(r['dataset'] != name or not r.get('replay_safe', False) for r in reads):
            continue
        bound = deepcopy(flow)
        known = {(r['variable'], r.get('cell'), r['line']) for r in reads}
        for source in bound.get('source_data', []):
            if (source.get('variable'), source.get('cell'), source.get('line')) in known:
                source['path'] = str(csv_path)
        for step in bound.get('replay_steps', []):
            if step.get('operation') == 'load' and (step.get('output'), step.get('cell'), step.get('line')) in known:
                step['path'] = str(csv_path)
        paths = {s.get('path') for s in bound.get('source_data', [])}
        if paths == {str(csv_path)}:
            candidates.append(bound)
    # Exact duplicate code artifacts do not create competing modeling contexts.
    deduplicated = {}
    for flow in candidates:
        signature = tuple(u.get('code') for u in flow.get('code_units', [])) or (flow.get('file'),)
        deduplicated.setdefault(signature, flow)
    candidates = list(deduplicated.values())
    if len(candidates) == 1:
        return {**candidates[0], 'files': [], 'strict_dataset_binding': True,
                'bound_csv_path': str(csv_path), 'bound_dataset_name': name, 'association_status': 'matched'}
    reason = '未能关联具体建模数据：存在多个独立流程。' if candidates else '未能关联具体建模数据：读取路径或参数尚未验证、存在动态路径或同流程读取多个 CSV。'
    empty = analyze_python_data_flow('')
    return {**empty, 'files': [], 'strict_dataset_binding': True, 'bound_csv_path': str(csv_path),
            'association_status': 'unassociated', 'association_reason': reason}


def sample_summary(graph, data_results):
    rows = {e['file']: e['result']['shape']['rows'] for e in data_results}
    return {'counting_unit': 'physical_csv_rows', 'unique_project_rows': None,
            'statement': '各 CSV 行数是文件内记录数；raw/train/test、派生产物可能重叠，不相加作为独立样本总数。',
            'datasets': [{'file': d['file'], 'rows': rows.get(d['file']), 'roles': d['roles'], 'parents': d['parents']} for d in graph['datasets']]}


def classify_dataset_role(name, result, association=None):
    """Infer a CSV's role from path, schema and verified read/write lineage.

    Strong artifact signatures take precedence over row-count heuristics.  An
    unassociated CSV remains unknown rather than being treated as training data.
    """
    association = association or {}
    existing = list(association.get('roles', []))
    normalized = {re.sub(r'[^a-z0-9]+', '', str(column).casefold())
                  for column in result.get('data_types', {})}
    path = name.replace('\\', '/').casefold()
    rows = int(result.get('shape', {}).get('rows') or 0)
    metric_columns = {'precision', 'recall', 'f1score', 'support'}
    importance_columns = {'importance', 'featureimportance', 'gain', 'weight'}
    feature_columns = {'feature', 'featurename', 'variable', 'column'}
    evidence = []

    if (len(metric_columns & normalized) >= 3 and rows <= 50) or 'classification_report' in path:
        role, confidence = 'model_evaluation_report', 'High'
        evidence.append('path/schema matches a compact classification metric report')
    elif (importance_columns & normalized and feature_columns & normalized) or 'feature_importance' in path:
        role, confidence = 'feature_importance_result', 'High'
        evidence.append('path/schema contains feature and importance fields')
    elif existing == ['train'] or re.search(r'(?:^|[/_.-])train(?:[/_.-]|$)', path):
        role, confidence = 'train', 'High'
        evidence.append('verified lineage or path identifies training data')
    elif existing == ['test'] or re.search(r'(?:^|[/_.-])test(?:[/_.-]|$)', path):
        role, confidence = 'test', 'High'
        evidence.append('verified lineage or path identifies test data')
    elif existing == ['validation'] or re.search(r'(?:^|[/_.-])(?:validation|valid|val)(?:[/_.-]|$)', path):
        role, confidence = 'validation', 'High'
        evidence.append('verified lineage or path identifies validation data')
    elif association.get('parents') or association.get('producers'):
        role, confidence = 'feature_engineering_result', 'Medium'
        evidence.append('static write lineage identifies a derived dataset')
    elif existing == ['raw'] or '/raw/' in '/' + path or re.search(r'(?:^|[/_.-])raw(?:[/_.-]|$)', path):
        role, confidence = 'raw', 'Medium'
        evidence.append('read lineage or path identifies raw data')
    else:
        role, confidence = 'unassociated', 'Low'
        evidence.append('no reliable path/schema/read-write role evidence')
    return {'primary_role': role, 'roles': [role], 'confidence': confidence,
            'evidence': evidence, 'uncertain': confidence == 'Low'}


def apply_dataset_role_policy(result, role_info):
    """Keep artifact evidence while preventing training-data rules from misfiring."""
    role = role_info.get('primary_role', 'unassociated')
    result['file_role'] = deepcopy(role_info)
    artifact_roles = {'model_evaluation_report', 'feature_importance_result', 'other_analysis_artifact'}
    if role not in artifact_roles:
        return result
    expected_artifact_rules = {
        'DQ_VERY_SMALL_DATASET', 'DQ_HIGH_CORRELATION', 'DQ_HIGH_CARDINALITY',
        'DQ_POTENTIAL_ID', 'DQ_POTENTIAL_INDEX_COLUMN', 'DQ_CONSTANT_COLUMN',
        'DQ_SINGLE_CATEGORY', 'DQ_EXTREME_SKEWNESS', 'DQ_POTENTIAL_EXTREME_VALUES',
        'DQ_DISCRETE_DISTRIBUTION_REVIEW',
    }
    for issue in result.get('issues', []):
        issue['artifact_role'] = role
        if issue.get('rule_id') not in expected_artifact_rules:
            continue
        issue['action'] = 'Informational'
        issue['score_impact'] = 0
        issue['severity'] = 'Low'
        issue['issue_nature'] = 'Artifact Observation'
        issue['deduplication_key'] = (
            f"artifact-index:{issue.get('column')}" if issue.get('rule_id') in {
                'DQ_HIGH_CARDINALITY', 'DQ_POTENTIAL_ID', 'DQ_POTENTIAL_INDEX_COLUMN'
            } else issue.get('deduplication_key')
        )
        if issue.get('rule_id') == 'DQ_VERY_SMALL_DATASET':
            issue['message'] = '结果汇总表行数较少，符合其文件角色；这不是训练样本量证据。'
            issue['recommendation'] = '按结果表结构核对指标或特征条目，不要将行数解释为训练集规模。'
        elif issue.get('rule_id') == 'DQ_HIGH_CORRELATION':
            issue['type'] = 'Correlated Report Metrics Observation'
            issue['message'] = '结果表指标之间相关；这些列不是已确认的建模输入特征。'
            issue['recommendation'] = '仅作为报告指标关系展示，不按特征共线性处理。'
    return result


def scoped_dataset_consistency(data_results, graph):
    """Compare only sibling artifacts/shared consumers, not raw versus transformed schemas."""
    from data_analyzer import analyze_dataset_consistency
    from itertools import combinations
    metadata = {d['file']: d for d in graph['datasets']}
    results, scope = [], []
    for left, right in combinations(sorted(data_results, key=lambda e: e['file']), 2):
        a, b = metadata[left['file']], metadata[right['file']]
        direct_parent = a['file'] in b['parents'] or b['file'] in a['parents']
        siblings = bool(set(a['parents']) & set(b['parents']))
        shared_consumer = bool({r['file'] for r in a['loaders']} & {r['file'] for r in b['loaders']})
        artifact_roles = {'model_evaluation_report', 'feature_importance_result', 'other_analysis_artifact'}
        role_mismatch = bool(set(a.get('roles', [])) & artifact_roles or set(b.get('roles', [])) & artifact_roles)
        compare = not direct_parent and not role_mismatch and (siblings or shared_consumer)
        scope.append({'files': [a['file'], b['file']], 'compared': compare,
                      'reason': '同上游派生或同代码消费，列模式差异仍需复核' if compare else
                                '模型结果产物与训练/业务数据不要求相同列模式' if role_mismatch else
                                '原始与派生不要求同列模式' if direct_parent else '未确认同一数据层或共同消费，未强制比较列模式'})
        if compare:
            for issue in analyze_dataset_consistency([left, right]):
                results.append({**issue, 'verification_status': 'needs_review',
                                'recommendation': '确认两文件的用途和变换边界；该差异不代表已证实的业务错误。' + issue.get('recommendation', '')})
    graph['schema_comparison_scope'] = scope
    return results
