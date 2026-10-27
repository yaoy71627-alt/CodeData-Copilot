"""Shared presentation policy; internal IDs and transport errors stay intact."""
import re
from copy import deepcopy

PARTIAL_MESSAGE = 'AI Review 已完成部分解释，其余问题已使用静态分析结果补齐。'


def ai_outcome(ai):
    status = ai.get('status', 'disabled')
    if status in {'success', 'partial_success'}:
        return status
    return 'not_needed' if status in {'disabled', 'not_needed'} else 'failed'


def normalized_ai(ai, coverage=None):
    ai = deepcopy(ai)
    if coverage is not None:
        ai['validation_rejections'] = deepcopy(coverage.get('rejected_explanations', []))
    if coverage is not None and ai.get('status') not in {None, 'disabled', 'not_needed'} and not ai.get('previous_review_retained'):
        accepted = coverage.get('accepted_annotations', {})
        complete = sum(all(fields.get(k) for k in ('cause', 'potential_impact', 'recommendation')) for fields in accepted.values())
        total = coverage.get('raw_finding_count', 0)
        ai['accepted_findings'] = complete
        ai['total_findings'] = total
        if complete < total and ai.get('status') in {'success', 'partial_success'}:
            ai['status'] = 'partial_success' if complete else 'invalid_ai_response'
            ai['reason'] = PARTIAL_MESSAGE if complete else 'AI 内容校验未通过；已保留完整静态报告。'
    ai['outcome'] = ai_outcome(ai)
    if ai['outcome'] == 'failed':
        ai['failure_code'] = ai.get('status', 'unknown')
    return ai


def review_notice(enhancement, warnings=(), error=None):
    """One event notice plus deduplicated diagnostic details, rebuilt each render."""
    ai, knowledge = enhancement.get('ai', {}), enhancement.get('knowledge', {})
    outcome = ai_outcome(ai)
    details = list(dict.fromkeys(str(x) for x in [ai.get('reason'), knowledge.get('reason'), error, *ai.get('validation_notes', []), *warnings] if x))
    if outcome == 'partial_success':
        return {'level': 'warning', 'message': PARTIAL_MESSAGE, 'details': details}
    if outcome == 'failed':
        return {'level': 'warning', 'message': 'AI 增强未完成，静态报告可正常导出：' + ai.get('reason', '原因未记录。'), 'details': details}
    if knowledge.get('status') not in {None, 'success', 'empty'}:
        return {'level': 'warning', 'message': '知识库增强未完成，基础分析结果已保留。', 'details': details}
    # Historical AI warnings must not survive a successful rerun.
    other = [w for w in warnings if is_analysis_warning(w)]
    return {'level': 'warning', 'message': other[0] if len(other) == 1 else '基础分析已完成，部分文件或导出步骤需要复核，请查看详细信息。', 'details': details} if other else None


def display_id_map(findings, problems):
    return {'findings': {fid: f'F{i:02d}' for i, fid in enumerate(sorted(f['id'] for f in findings), 1)},
            'problems': {pid: f'P{i:02d}' for i, pid in enumerate(sorted(p['id'] for p in problems), 1)}}


def display_text(text, ids):
    mapping = {**ids.get('findings', {}), **ids.get('problems', {})}
    return re.sub(r'\b[FP]-[0-9a-f]+(?:-\d+)?\b', lambda m: mapping.get(m[0], m[0]), str(text or '')).replace('P- 编号', 'P 编号').replace('F- 编号', 'F 编号').replace('P- 问题编号', 'P 问题编号').replace('F- 证据', 'F 证据')


def is_analysis_warning(message):
    """Keep actual static/import/export faults when an AI job is refreshed."""
    return bool(re.search(r'：(?:解析失败|代码质量分析失败|数据质量分析失败|数据集为空)|^PDF 报告生成失败：', str(message)))
