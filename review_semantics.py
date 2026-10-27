"""Conservative, scoped checks on explanation claims, independent of model transport."""
import json
import re


def explanation_text_issue(text):
    """Return a reason when prose is visibly truncated or is a code fragment."""
    if not isinstance(text, str) or not text.strip():
        return '解释字段为空。'
    value = text.strip()
    if value.startswith((']', ')', '}', ',', '，', '.', '=')):
        return '解释以孤立的代码或标点片段开头。'
    if re.match(r'^(?:并|或|且|以及)\s*(?!非|不是)', value):
        return '解释以缺少上文的连接词开头。'
    if re.match(r'^(?:and|or)\b', value, re.I):
        return '解释以缺少上文的连接词开头。'
    if re.search(r'(?:[,，:：;；]|[-+*/=]|\b(?:and|or|because|with|to))\s*$', value, re.I):
        return '解释在连接词、运算符或未完成标点处截断。'
    if re.search(r'(?:以及|并且|而且|因为|例如|包括|建议|需要|通过|导致|用于|将|对|的)\s*[。.!?]?$' , value):
        return '解释缺少完整的操作对象或结论。'
    pairs = [('(', ')'), ('[', ']'), ('{', '}')]
    if any(value.count(left) != value.count(right) for left, right in pairs):
        return '解释包含未闭合或孤立的括号。'
    if value.count('`') % 2:
        return '解释包含未闭合的代码标记。'
    if re.match(r'^[A-Za-z_]\w*\s*\]|^[A-Za-z_]\w*\s*[,)]\s*[A-Za-z_]\w*\s*=', value):
        return '解释看起来是代码中间片段。'
    meaningful = re.sub(r'[`\W_]+', '', value, flags=re.UNICODE)
    if len(meaningful) < 4:
        return '解释没有形成可理解的完整陈述。'
    return None


def explanation_entry_issue(entry):
    """Validate all narrative fields without relying on length or keywords alone."""
    if not isinstance(entry, dict):
        return '解释记录不是对象。'
    for key in ('cause', 'potential_impact', 'recommendation'):
        reason = explanation_text_issue(entry.get(key))
        if reason:
            return f'{key}：{reason}'
    return None


def family(finding):
    text = ' '.join(str(finding.get(k, '')) for k in ('rule_id', 'type', 'category')).lower()
    for name, pattern in (
        ('dropna', r'dropna|sample.loss'), ('missing', r'missing|缺失'),
        ('no_cv', r'no_cross_validation|cross.validation.coverage'),
        ('repeat_test', r'repeated.*(?:test|evaluation)|test.*(?:reuse|repeated|overuse)|重复.*评'),
        ('preprocess', r'preprocess|leakage'),
    ):
        if re.search(pattern, text):
            return name
    return 'other'


def scoped_context(finding, facts):
    """Only attach a measured operation when its source location agrees."""
    loc = finding.get('location', {})
    impacts = []
    if family(finding) == 'dropna':
        for item in facts.get('quantified_impacts', []):
            if (item.get('flow_file', item.get('file')) == loc.get('file') and
                all(loc.get(k) is None or item.get(k) == loc.get(k) for k in ('cell', 'line')) and
                item.get('quantification') == 'Exact' and
                item.get('operation') == 'dropna'):
                impacts.append({k: item[k] for k in ('file', 'cell', 'line', 'before', 'after', 'removed', 'removal_rate') if k in item})
    context = {'measured_operations': impacts} if impacts else {}
    models = [facts.get('duplicate_split_analysis', {}).get('modeling_data', {})]
    models.extend(d.get('modeling_data', {}) for d in facts.get('datasets', []))
    matching = [m for m in models if m.get('row_reconstruction') == 'Exact' and m.get('flow_file') == loc.get('file')]
    if finding.get('origin') == 'code' and matching:
        candidates = [{k: m[k] for k in ('raw_rows', 'cleaned_rows', 'train_rows', 'test_rows') if k in m} for m in matching]
        if all(m == candidates[0] for m in candidates):
            context['reconstructed_split'] = candidates[0]
    column = loc.get('column')
    if column:
        matching_datasets = [
            dataset for dataset in facts.get('datasets', [])
            if dataset.get('file') == loc.get('file')
        ]
        if len(matching_datasets) == 1:
            profile = (matching_datasets[0].get('numeric_statistics') or {}).get(column)
            if isinstance(profile, dict):
                context['field_profile'] = {
                    key: profile[key]
                    for key in ('minimum', 'maximum', 'min', 'max', 'q1', 'q3', 'median')
                    if profile.get(key) is not None
                }
    return context


def grounded_numbers(text, finding, facts):
    """Numbers must come from this evidence, not an unrelated dataset/finding."""
    if re.search(r'(?:准确率|性能|accuracy|auc|precision|recall|得分).{0,10}\d', text, re.I):
        return False
    evidence = ' '.join(str(finding.get(k, '')) for k in ('evidence', 'source_snippet', 'detected_count', 'detected_rate'))
    context = scoped_context(finding, facts)
    number_pattern = r'(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?'
    allowed = set(re.findall(number_pattern, evidence + json.dumps(context)))
    if re.search(r'\blog1p\b', text, re.I):
        # Mathematical domain boundary, not a project-data claim.
        allowed.add('-1')
    profile = context.get('field_profile', {})
    minimum = profile.get('minimum', profile.get('min'))
    if (isinstance(minimum, (int, float)) and minimum < -1
            and re.search(r'(?:\u5c0f\u4e8e|\u4f4e\u4e8e|less\s+than|[<>]=?)\s*-1', text, re.I)):
        # This is the verified log1p-domain guard for a field that actually
        # contains values below the boundary, not an observed record count.
        allowed.add('-1')
    for label, key in ((r'Cell|单元格', 'cell'), (r'Line|第', 'line')):
        for value in re.findall(r'(?:' + label + r')\s*(\d+)', text, re.I):
            if finding.get('location', {}).get(key) == int(value):
                allowed.add(value)
    percentages = set(re.findall(r'(\d+(?:\.\d+)?)\s*%', evidence))
    rates = [finding.get('detected_rate')] + [m.get('removal_rate') for m in context.get('measured_operations', [])]
    for rate in rates:
        if isinstance(rate, (int, float)):
            percentages.update(format(rate * 100, '.' + str(n) + 'f').rstrip('0').rstrip('.') if n else format(rate * 100, '.0f') for n in (0, 1, 2))
    if any(n not in percentages for n in re.findall(r'(\d+(?:\.\d+)?)\s*%', text)):
        return False
    allowed.update(percentages)
    if re.search(r'[一二三四五六七八九十百千万]+(?:行|条|组|个|倍|百分点)', text):
        return False
    # A bound/min/max is not evidence for an exact number of outlier records.
    outlier_count_patterns = (
        r'(?:IQR|\u533a\u95f4\u5916|\u5f02\u5e38|\u6781\u7aef|\u79bb\u7fa4|outliers?).{0,20}?(\d+)\s*(?:\u6761|\u884c|records?|rows?)',
        r'(\d+)\s*(?:\u6761|\u884c|records?|rows?).{0,20}?(?:IQR|\u533a\u95f4\u5916|\u5f02\u5e38|\u6781\u7aef|\u79bb\u7fa4|outliers?)',
    )
    claimed_outlier_counts = {
        value for pattern in outlier_count_patterns for value in re.findall(pattern, text, re.I)
    }
    if claimed_outlier_counts:
        detected = finding.get('detected_count')
        if detected is None or claimed_outlier_counts != {str(detected)}:
            return False
    numbers = re.findall(number_pattern, text)
    if any(n not in allowed for n in numbers):
        return False
    split = context.get('reconstructed_split', {})
    for pattern, key in ((r'(?:训练集|train(?:ing)?(?: set)?)\D{0,8}(\d+)\s*(?:行|rows?)', 'train_rows'),
                         (r'(?:测试集|test(?: set)?)\D{0,8}(\d+)\s*(?:行|rows?)', 'test_rows')):
        for number in re.findall(pattern, text, re.I):
            if str(split.get(key)) != number:
                return False
    for item in context.get('measured_operations', []):
        for pattern, key in ((r'(?:删除|移除|丢弃|removed?)\D{0,8}(\d+)\s*(?:行|rows?)', 'removed'),
                             (r'(?:处理前|删除前|before)\D{0,8}(\d+)\s*(?:行|rows?)', 'before'),
                             (r'(?:处理后|删除后|剩余|保留|after)\D{0,8}(\d+)\s*(?:行|rows?)', 'after')):
            if any(str(item.get(key)) != n for n in re.findall(pattern, text, re.I)):
                return False
    return True


def annotation_mismatch(entry, finding, facts):
    """Reject demonstrably wrong bindings; a generic cautious explanation is allowed.

    The cause is checked against the detected event's role, and explicit field,
    location and measured-operation claims are checked against its source scope.
    Merely mentioning another operation in a proposed remedy is not a mismatch.
    """
    cause = entry.get('cause', '')
    # Explanations may contrast the actual cause with an unrelated operation.
    # Do not treat a negated comparison as an assertion of that other cause.
    cause_claims = '。'.join(clause for clause in re.split(r'[。；;\n]', cause)
                           if not re.search(r'不能替代|无法替代|不等于|并不代表|不意味着|并非|不是.{0,12}原因', clause))
    fields = {
        key: str(entry.get(key, ''))
        for key in ('cause', 'potential_impact', 'recommendation')
    }
    text = '\n'.join(fields.values())
    task = str((facts.get('task_type') or {}).get('type') or '').casefold()
    recommendation = str(entry.get('recommendation', ''))
    if task == 'classification' and re.search(
        r'(?:建议|使用|采用|改用|try|use|recommend).{0,20}(?:HuberRegressor|LinearRegression|Ridge|Lasso|ElasticNet|\bR\s*[²2]\b|\bMAE\b|\bMSE\b)',
        recommendation, re.I,
    ):
        return '建议的模型或评价指标属于回归任务，与已识别的分类任务不匹配。'
    if task == 'regression' and re.search(
        r'(?:建议|使用|采用|改用|try|use|recommend).{0,20}(?:LogisticRegression|precision|recall|f1.score|roc.auc)',
        recommendation, re.I,
    ):
        return '建议的模型或评价指标属于分类任务，与已识别的回归任务不匹配。'
    allowed_claims = json.dumps({
        'finding': finding, 'verified_context': scoped_context(finding, facts)
    }, ensure_ascii=False, default=str).casefold()
    for unit in re.findall(r'(?<![A-Za-z_])(?:kW|hp|cc|kg|km|miles?|USD|CNY|千瓦|马力)(?![A-Za-z_])', text, re.I):
        if unit.casefold() not in allowed_claims:
            return '解释添加了原始证据未提供的数值单位。'
    if re.search(r'事故车|豪华车|真实车市|市场采样|医疗场景|金融场景', text, re.I):
        return '解释添加了项目证据未提供的业务背景。'
    if finding.get('rule_id') == 'CODE_ONEHOT_SPARSE_API' and re.search(r'AttributeError', text, re.I):
        return 'OneHotEncoder 旧参数的不兼容异常类型被错误描述；未采用该解释。'
    kind = family(finding)
    foreign_causes = {
        'missing': r'(?:未|没有|缺少|缺乏|未使用|未采用).{0,8}(?:交叉验证|cross.validation)|no cross.validation',
        'repeat_test': r'(?:数据采集|数据收集|下载|data collection).{0,8}(?:失败|错误|failure|failed)',
        'no_cv': r'(?:重复|反复|多次).{0,12}(?:accuracy_score|准确率|测试集|评估)|(?:accuracy_score).{0,12}(?:重复|多次)',
    }
    if kind in foreign_causes and re.search(foreign_causes[kind], cause_claims, re.I):
        return '解释描述的原因与该问题的检测类型不符，已改用对应结构化事实。'
    loc = finding.get('location', {})
    column = loc.get('column')
    claim_text = text
    field_labels = {
        'cause': '原因', 'potential_impact': '潜在影响', 'recommendation': '修复建议',
    }
    fields_by_dataset = {
        dataset.get('file'): {str(value) for value in dataset.get('column_names', [])}
        for dataset in facts.get('datasets', [])
    }
    known_fields = set().union(*fields_by_dataset.values()) if fields_by_dataset else set()
    if column:
        # Missing-value explanations are scoped to their own field across all
        # three narrative sections.  A foreign field is not an acceptable
        # subject merely because it appeared in another Finding in the batch.
        if kind == 'missing':
            local_fields = fields_by_dataset.get(loc.get('file')) or known_fields
            for key, value in fields.items():
                for foreign in local_fields - {str(column)}:
                    if re.search(r'(?<![A-Za-z_])' + re.escape(foreign) + r'(?![A-Za-z_])', value, re.I):
                        return f'{field_labels[key]}指向了其他字段 {foreign}，与当前 Finding 的目标字段 {column} 不一致。'
    if kind != 'missing':
        missing_claim = r'(?:\u7f3a\u5931|\u7a7a\u503c|missing|imput|\u586b\u8865|is_[A-Za-z_]\w*_missing)'
        for key, value in fields.items():
            if re.search(missing_claim, value, re.I) and (
                any(re.search(r'(?<![A-Za-z_])' + re.escape(name) + r'(?![A-Za-z_])', value, re.I)
                    for name in known_fields)
                or re.search(r'is_[A-Za-z_]\w*_missing|imput|\u586b\u8865', value, re.I)
            ):
                return f'{field_labels[key]}讨论了字段缺失处理，与当前 Finding 的问题类型不一致。'
    profile = scoped_context(finding, facts).get('field_profile', {})
    minimum = profile.get('minimum', profile.get('min'))
    if isinstance(minimum, (int, float)) and minimum < -1 and re.search(r'\blog1p\s*\(?', recommendation, re.I):
        prerequisite = re.search(
            r'(?:\u5148|\u4e4b\u524d|before|after|\u524d\u7f6e|\u6761\u4ef6|\u4ec5\u5bf9).{0,30}'
            r'(?:\u5e73\u79fb|shift|clip|\u622a\u65ad|\u8fc7\u6ee4|\u5254\u9664|\u4fee\u6b63|\u5904\u7406|\u6838\u5b9e|\u6821\u9a8c|\u9a8c\u8bc1|>=?\s*-?1|\u975e\u8d1f)',
            recommendation, re.I,
        )
        if not prerequisite:
            return (
                f'修复建议直接对最小值为 {minimum} 的字段使用 log1p，'
                '但没有先处理或校验小于 -1 的取值。'
            )
    known_files = {f.get('location', {}).get('file') for f in facts.get('findings', [])}
    for file in re.findall(r'[\w./-]+\.(?:py|ipynb|csv)\b', claim_text):
        if file in known_files and file != loc.get('file') and not str(loc.get('file', '')).endswith('/' + file):
            return '解释的文件位置与该 Finding 不一致。'
    for label, key in ((r'(?:Cell|单元格)', 'cell'), (r'(?:Line|第)\s*', 'line')):
        for number in re.findall(label + r'\s*(\d+)\s*(?:行)?', cause, re.I):
            if loc.get(key) is not None and int(number) != loc[key]:
                return '解释的代码位置与该 Finding 不一致。'
    measured = scoped_context(finding, facts).get('measured_operations', [])
    if measured and re.search(r'(?:无法|不能|尚未|未能|难以).{0,10}(?:量化|计算|确定|验证).{0,10}(?:删除|移除|丢弃|行数|数量|样本损失)|(?:删除|移除|丢弃).{0,10}(?:无法|不能|未知|不明)', text):
        return '该操作已有精确行数重建，未采用无法量化的相反解释。'
    model = facts.get('duplicate_split_analysis', {}).get('modeling_data', {})
    if (model.get('flow_file') in {None, loc.get('file')} and
        model.get('row_reconstruction') == 'Exact') and kind in {'preprocess', 'repeat_test'}:
        if re.search(r'(?:未提供|未知|其他|额外).{0,12}(?:划分文件|split.files)|(?:无法|未能).{0,8}重建.{0,6}(?:划分|训练|测试)', text, re.I):
            return '已重建当前建模划分，未采用关于未知划分文件的推测。'
    return None
