"""Regression for scoped explanations, one event notice, and stable display IDs."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pypdf import PdfReader

from consistency_validator import (build_analysis_facts, build_review_coverage,
    render_structured_review, validate_agent_consistency, ReviewText,
    validate_report_consistency)
from report import build_report_data, build_markdown_report, build_pdf_report
from ui_review_helpers import (PARTIAL_MESSAGE, review_notice, display_id_map,
                               display_text, normalized_ai)


@pytest.fixture
def bundle():
    code = [{'file': 'titanic.ipynb', 'cell': 5, 'issues': [
        {'rule_id': 'CODE_DROPNA_SAMPLE_LOSS', 'type': 'Missing Value Handling Risk',
         'line': 7, 'severity': 'Medium', 'evidence': 'model_data.dropna()',
         'source_snippet': 'model_data.dropna()', 'recommendation': '核对缺失机制。'},
        {'rule_id': 'CODE_NO_CROSS_VALIDATION', 'type': 'No Cross-validation Detected',
         'line': 20, 'severity': 'Medium', 'evidence': '仅使用单次划分；accuracy_score 评价。'},
        {'rule_id': 'CODE_TEST_SET_OVERUSE', 'type': 'Repeated Test-set Evaluation Requires Review',
         'line': 23, 'severity': 'High', 'evidence': '循环使用测试集评估模型。'},
        {'rule_id': 'CODE_PREPROCESS_BEFORE_SPLIT', 'type': 'Data Leakage Risk',
         'line': 13, 'severity': 'High', 'evidence': 'scaler.fit_transform(X)'}]}]
    data = [{'file': 'train.csv', 'result': {'shape': {'rows': 916, 'columns': 2},
       'data_types': {'Sex': 'object', 'Age': 'float'}, 'issues': [
         {'rule_id': 'DQ_MISSING_VALUES', 'type': 'Missing Values', 'category': 'Missing Values',
          'column': 'Sex', 'severity': 'Medium', 'evidence': 'Sex 存在缺失值。', 'count': 3}]}}]
    facts = build_analysis_facts(code, data, {'split': {'detected': True}},
       quantified_impacts=[{'operation': 'dropna', 'flow_file': 'titanic.ipynb',
          'cell': 5, 'line': 7, 'quantification': 'Exact', 'before': 916, 'after': 647, 'removed': 269}],
       duplicate_split_analysis={'modeling_data': {'row_reconstruction': 'Exact',
          'flow_file': 'titanic.ipynb', 'train_rows': 485, 'test_rows': 162}})
    return code, data, facts


def entry(finding, cause):
    return {'finding_id': finding['id'], 'cause': cause,
            'potential_impact': '实际影响需要运行对照确认。',
            'recommendation': '结合原始证据核对当前处理路径。'}


def by_rule(facts, rule):
    return next(f for f in facts['findings'] if f['rule_id'] == rule)


@pytest.mark.parametrize('rule,cause', [
    ('DQ_MISSING_VALUES', '没有交叉验证，无法衡量不同划分的波动。'),
    ('DQ_MISSING_VALUES', 'Age 字段缺失，输入可能不完整。'),
    ('CODE_TEST_SET_OVERUSE', '数据采集失败，导致无法获得可靠样本。'),
    ('CODE_NO_CROSS_VALIDATION', '重复调用 accuracy_score 导致评估偏差。'),
    ('CODE_DROPNA_SAMPLE_LOSS', '无法量化删除行数，需要用户提供数据。'),
    ('CODE_DROPNA_SAMPLE_LOSS', '无法量化样本损失。'),
    ('CODE_PREPROCESS_BEFORE_SPLIT', '可能与未提供的其他划分文件重复。'),
    ('CODE_NO_CROSS_VALIDATION', '问题在 titanic.ipynb 的 Line 23。'),
])
def test_misbound_explanation_is_rejected_and_filled(bundle, rule, cause):
    facts = bundle[2]
    finding = by_rule(facts, rule)
    review = render_structured_review({'explanations': [entry(finding, cause)]}, facts)
    assert cause not in review
    assert finding['id'] not in review.coverage['accepted_annotations']
    assert review.coverage['rejected_explanations'][0]['finding_id'] == finding['id']
    assert review.coverage['ai_explained'] == 0
    assert all(f['id'] in review for f in facts['findings'])
    assert '结构化事实补齐' in review
    if rule == 'CODE_DROPNA_SAMPLE_LOSS':
        assert '删除 269 行' in review


@pytest.mark.parametrize('rule,cause', [
    ('DQ_MISSING_VALUES', 'Sex 字段存在缺失，实际影响取决于模型是否使用该字段。'),
    ('CODE_TEST_SET_OVERUSE', '反复查看测试集评价并选择模型，可能使测试集参与模型选择。'),
    ('CODE_NO_CROSS_VALIDATION', '仅使用单次划分，尚不能观察不同折之间的评价波动。'),
    ('CODE_NO_CROSS_VALIDATION', '重复调用 accuracy_score 不能替代交叉验证。'),
    ('CODE_DROPNA_SAMPLE_LOSS', 'dropna 删除 269 行，处理前 916 行，处理后 647 行。'),
    ('CODE_PREPROCESS_BEFORE_SPLIT', '预处理器在划分前拟合，可能使用测试部分的分布信息。'),
    ('CODE_PREPROCESS_BEFORE_SPLIT', 'Cell 5 的 Line 13 在划分前拟合预处理器，可能使用测试部分的信息。'),
    ('CODE_PREPROCESS_BEFORE_SPLIT', '建议核对建模输入与预处理的执行路径。'),
])
def test_normal_explanations_survive(bundle, rule, cause):
    facts = bundle[2]
    finding = by_rule(facts, rule)
    review = render_structured_review({'explanations': [entry(finding, cause)]}, facts)
    checked, _ = validate_agent_consistency(review, facts)
    assert cause in checked
    assert checked.coverage['ai_explained'] == 1


def test_numbers_from_another_finding_cannot_be_borrowed(bundle):
    facts = bundle[2]
    finding = by_rule(facts, 'CODE_DROPNA_SAMPLE_LOSS')
    for cause in ('dropna 删除 647 行。', 'dropna 删除 9999 行。', '训练集有 647 行。', '准确率为 269%。', '处理前 647 行。', '删除比例为 269%。'):
        review = render_structured_review({'explanations': [entry(finding, cause)]}, facts)
        assert cause not in review
        assert review.coverage['ai_explained'] == 0
    foreign = deepcopy(facts)
    foreign['quantified_impacts'][0]['flow_file'] = 'other.ipynb'
    review = render_structured_review({'explanations': [entry(finding, '删除 269 行。')]}, foreign)
    assert '删除 269 行。' not in review


def test_tampered_typed_review_is_revalidated(bundle):
    facts = bundle[2]
    finding = by_rule(facts, 'DQ_MISSING_VALUES')
    bad = entry(finding, '没有交叉验证。')
    coverage = build_review_coverage(facts['findings'], {finding['id']: {k: bad[k] for k in ('cause', 'potential_impact', 'recommendation')}})
    forged = ReviewText('旧的不可信文本。没有交叉验证。', coverage,
                        {'status': 'success', 'reason': '旧状态'})
    checked, _ = validate_agent_consistency(forged, facts)
    assert '没有交叉验证' not in checked and '旧的不可信文本' not in checked
    assert checked.coverage['ai_explained'] == 0
    assert checked.ai_status['outcome'] == 'failed'


def make_report(bundle, status='partial_success'):
    code, data, facts = bundle
    finding = by_rule(facts, 'DQ_MISSING_VALUES')
    review = render_structured_review({'explanations': [entry(finding, 'Sex 字段缺失，需要核对处理路径。')]}, facts)
    review.ai_status = {'status': status, 'reason': '仅部分 Finding 获得解释。'}
    return validate_report_consistency(build_report_data(
       {'project_name': 'Titanic', 'python_file_count': 1, 'file_count': 2},
       code, data, review, [], '', {}, analysis_facts=facts,
       enhancement_status={'knowledge': {'status': 'success', 'reason': '已检索'}, 'ai': review.ai_status}))


def test_short_ids_unique_stable_and_exported_consistently(bundle):
    report = make_report(bundle)
    ids = report['display_ids']
    assert ids == display_id_map(list(reversed(report['findings'])), list(reversed(report['conclusions']['problems'])))
    assert len(set(ids['findings'].values())) == len(report['findings'])
    assert all(f['id'].startswith('F-') for f in report['findings'])
    md = build_markdown_report(report)
    pdf = build_pdf_report(report)
    assert pdf.startswith(b'%PDF')
    pdf_text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    view = report['view']['agent_report']
    for row in report['conclusions']['finding_rows']:
        for short in (row['display_id'], row['display_problem_id']):
            assert short in md and short in pdf_text and short in view
        assert row['id'] not in md and row['id'] not in pdf_text and row['id'] not in view
    assert report['enhancement_status']['ai']['outcome'] == 'partial_success'
    assert report['review_coverage']['ai_explained'] == 1
    assert report['conclusions']['counts']['total'] == len(report['conclusions']['problems'])


def test_one_problem_can_have_multiple_short_evidence_ids(bundle):
    facts = deepcopy(bundle[2])
    for f in facts['findings']:
        f['deduplication_key'] = 'one-event'
    from consistency_validator import group_findings
    problems = group_findings(facts['findings'])
    assert len(problems) == 1 and len(problems[0]['finding_ids']) == len(facts['findings'])
    ids = display_id_map(facts['findings'], problems)
    assert list(ids['problems'].values()) == ['P01']
    assert len(set(ids['findings'].values())) == len(facts['findings'])


@pytest.mark.parametrize('status,expected', [('success', None), ('partial_success', PARTIAL_MESSAGE),
                                         ('timeout', 'AI 增强未完成')])
def test_notice_policy(status, expected):
    notice = review_notice({'ai': {'status': status, 'reason': '本次原因'},
                            'knowledge': {'status': 'success'}},
                           ['旧的修正警告', '本次原因', '本次原因'], '旧错误')
    if expected is None:
        assert notice is None
    else:
        assert notice['message'].startswith(expected)
        assert len(notice['details']) == len(set(notice['details']))


@pytest.mark.parametrize('status,warning_count', [('partial_success', 1), ('success', 0), ('timeout', 1)])
def test_streamlit_home_single_notice_and_no_stale_rerun(bundle, status, warning_count):
    import app
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30)
    at.session_state['analyzed_files'] = {'code': '1', 'data': '1'}
    at.session_state['enhancement_status'] = {'ai': {'status': status, 'reason': '接口超时' if status == 'timeout' else '本次解释状态'},
                                            'knowledge': {'status': 'success'}}
    at.session_state['analysis_error'] = '旧的 AI 报告生成失败'
    at.session_state['analysis_warnings'] = ['已精确修正局部不可信结论', '本次解释状态']
    at.run()
    assert not at.exception
    # Ignore font resource diagnostics in the sidebar; the homepage event is one notice.
    homepage = [w.value for w in at.main.warning]
    assert len(homepage) == warning_count
    if status == 'partial_success':
        assert homepage == [PARTIAL_MESSAGE]
        assert not any('生成失败' in w for w in homepage)
    at.session_state['enhancement_status'] = {'ai': {'status': 'success'}, 'knowledge': {'status': 'success'}}
    at.run()
    assert not at.main.warning and not at.exception


def test_success_retains_real_static_warning():
    notice = review_notice({'ai': {'status': 'success'}}, ['analysis.py：代码质量分析失败（解析错误）。'])
    assert notice and '代码质量分析失败' in notice['message']


def test_known_split_counts_survive_without_becoming_deleted_rows(bundle):
    facts = bundle[2]
    finding = by_rule(facts, 'CODE_PREPROCESS_BEFORE_SPLIT')
    cause = '已重建训练集 485 行、测试集 162 行，预处理的拟合范围仍需核查。'
    review = render_structured_review({'explanations': [entry(finding, cause)]}, facts)
    assert cause in review and review.coverage['ai_explained'] == 1


def test_same_finding_evidence_is_not_duplicated_after_final_validation(bundle):
    facts = bundle[2]
    finding = by_rule(facts, 'DQ_MISSING_VALUES')
    review = render_structured_review({'explanations': [entry(finding, 'Sex 字段缺失。'), entry(finding, 'Sex 字段缺失。')]}, facts)
    checked, _ = validate_agent_consistency(review, facts)
    for f in facts['findings']:
        assert str(checked).count('### ' + f['id'] + ' ·') == 1
    assert checked.coverage['ai_explained'] == 1
