"""Four pinned cases: reviewed truth, partial Mock AI, exports and five UI pages."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import agent
import app
from consistency_validator import conclusion_summary_lines, finding_summary_line
from regression_benchmark import evaluate_case
from report import build_markdown_report

ROOT = Path(__file__).parent
TRUTH = json.loads((ROOT / 'fixtures/batch_d/truth.json').read_text(encoding='utf-8'))


def files_for_case(case, root):
    if case in ('titanic', 'cardekho'):
        expected = TRUTH['inputs'][case]
        prefix = 'CODEDATA_' + case.upper()
        sources = [os.environ.get(prefix + '_NOTEBOOK'), os.environ.get(prefix + '_CSV')]
        if not any(sources):
            pytest.skip(f'Original {case} files not configured; use scripts/run_batch_d.py for strict four-case acceptance')
        assert all(sources), f'{prefix}_NOTEBOOK and {prefix}_CSV must both be configured'
        files = []
        for alias, source in zip(expected, sources):
            path = Path(source)
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[alias], 'Case hash differs from the reviewed snapshot'
            local = root / alias
            local.write_bytes(path.read_bytes())
            files.append((alias, local))
        return files
    old, new = ROOT / 'fixtures/batch_b' / case, ROOT / 'fixtures/batch_c/telco'
    if case == 'ibm':
        return [(name, old / name) for name in ('employee-attrition.ipynb', 'emp_attrition.csv')]
    return [(p.relative_to(old).as_posix(), p) for p in sorted(old.rglob('*.py'))] + [
        ('README.md', old / 'README.md'), ('Makefile', new / 'Makefile')
    ] + [(p.relative_to(new).as_posix(), p) for p in sorted(new.rglob('*.csv'))]


@pytest.fixture(scope='module', params=['titanic', 'cardekho', 'ibm', 'telco'])
def d_case(request, tmp_path_factory):
    case = request.param
    files = files_for_case(case, tmp_path_factory.mktemp('batch_d_' + case))
    before = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files}
    calls = []
    def create(**kwargs):
        raw = kwargs['messages'][1]['content'].split('结构化 Finding 与 analysis_facts（唯一事实依据）：\n', 1)[1]
        compact, _ = json.JSONDecoder().raw_decode(raw)
        # Deliberately omit all but one finding in each response. A bounded retry
        # should retain that accepted item and resend only the still-missing IDs.
        calls.append({'ids': [f['id'] for f in compact['findings']],
                      'input_characters': sum(len(m['content']) for m in kwargs['messages'])})
        first = compact['findings'][0]
        payload = {'explanations': [{'finding_id': first['id'],
                    'cause': '此项来自原始静态证据，需按所列位置核对建模路径。',
                    'potential_impact': '影响范围尚需运行对照或业务确认。',
                    'recommendation': '先核对证据和上下游，再在隔离环境验证改动。'}]}
        return NS(choices=[NS(message=NS(content=json.dumps(payload, ensure_ascii=False)))])
    client = NS(chat=NS(completions=NS(create=create)))
    state = NS()
    with patch.object(app.st, 'session_state', state), patch.object(agent, 'get_client', return_value=client):
        app.analyze_project_files(files, generate_ai=True, project_metadata={'project_name': case})
    assert before == {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files}
    result = evaluate_case(state.report_data['findings'], TRUTH['cases'][case])
    output = os.environ.get('CODEDATA_BATCH_D_ARTIFACT_DIR')
    if output:
        out = Path(output) / case
        out.mkdir(parents=True, exist_ok=True)
        for name, value in [('benchmark.json', result), ('report.json', state.report_data),
                            ('run.json', {'case': case, 'input_sha256': before, 'mock_calls': calls,
                                          'enhancement_status': state.enhancement_status})]:
            (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        (out / 'report.md').write_text(state.report_markdown)
        if state.report_pdf:
            (out / 'report.pdf').write_bytes(state.report_pdf)
    return case, state, result, calls


def test_four_case_reviewed_truth_subset(d_case):
    _, state, result, _ = d_case
    assert result['fn'] == 0, result['missed']
    assert result['fp'] == 0, result['negative_assertions']
    assert state.report_error is None
    assert state.report_pdf.startswith(b'%PDF')
    assert state.enhancement_status['ai']['status'] == 'partial_success'


def test_mock_omissions_retry_only_unaccepted_findings(d_case):
    _, state, _, calls = d_case
    accepted = set()
    for call in calls:
        assert accepted.isdisjoint(call['ids'])
        accepted.add(call['ids'][0])
    assert max(c['input_characters'] for c in calls) <= 14000
    settings = agent.load_settings()
    assert len(calls) <= settings.max_batches * (settings.retry_count + 1)
    final = state.report_data
    assert final['review_coverage']['missing_problem_ids'], 'Mock must actually exercise deterministic fallback'
    for f in final['findings']:
        assert f['id'] in final['ai_review']
    assert state.patch_text == final['patch_text'] == ''


def test_four_case_canonical_markdown_and_pdf(d_case):
    _, state, _, _ = d_case
    final = state.report_data
    assert state.report_markdown == build_markdown_report(final)
    reader = PdfReader(BytesIO(state.report_pdf))
    assert len(reader.pages) > 1
    compact = lambda s: ''.join(s.split())
    text = compact('\n'.join(p.extract_text() for p in reader.pages))
    for line in conclusion_summary_lines(final) + [finding_summary_line(f) for f in final['conclusions']['finding_rows']]:
        assert line in state.report_markdown
        assert compact(line) in text


@pytest.mark.parametrize('page', ['Code Analysis', 'Data Quality', 'AI Review', 'Repair', 'Report'])
def test_four_case_streamlit_pages(d_case, page):
    case, state, _, _ = d_case
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=40)
    for key, value in vars(state).items():
        at.session_state[key] = value
    at.session_state['active_page'] = page
    at.run()
    assert not at.exception
    assert at.session_state['report_data']['final_fingerprint'] == state.report_data['final_fingerprint']
    assert at.session_state['patch_text'] == state.report_data['patch_text']
    if page in ('AI Review', 'Report'):
        for line in conclusion_summary_lines(state.report_data):
            assert line in [caption.value for caption in at.caption]
    if page == 'Report':
        downloads = [element.proto for element in at.get('download_button')]
        assert any('PDF' in item.label for item in downloads)
        assert any('Markdown' in item.label for item in downloads)
    output = os.environ.get('CODEDATA_BATCH_D_ARTIFACT_DIR')
    if output:
        (Path(output) / case / ('ui-' + page.replace(' ', '-') + '.json')).write_text(json.dumps({
            'page': page, 'exceptions': len(at.exception), 'fingerprint': state.report_data['final_fingerprint'],
            'method': 'Streamlit AppTest; not a browser click/download or live API test'}, ensure_ascii=False, indent=2))


@pytest.mark.parametrize('fault', ['missing_key', 'timeout', 'invalid_json', 'knowledge_and_api_failure'])
def test_four_real_cases_survive_enhancement_faults(d_case, tmp_path, monkeypatch, fault):
    from contextlib import ExitStack
    case, baseline, _, _ = d_case
    files = files_for_case(case, tmp_path)
    state = NS()
    def create(**kwargs):
        if fault == 'timeout':
            raise TimeoutError('private diagnostic details must not enter reports')
        return NS(choices=[NS(message=NS(content='not valid JSON'))])
    expected = {'missing_key': 'missing_api_key', 'timeout': 'timeout',
                'invalid_json': 'invalid_ai_response', 'knowledge_and_api_failure': 'missing_api_key'}[fault]
    with ExitStack() as stack:
        stack.enter_context(patch.object(app.st, 'session_state', state))
        if fault in ('missing_key', 'knowledge_and_api_failure'):
            monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
        else:
            stack.enter_context(patch.object(agent, 'get_client', return_value=NS(chat=NS(completions=NS(create=create)))))
        if fault == 'knowledge_and_api_failure':
            stack.enter_context(patch.object(agent, 'retrieve_relevant_knowledge', side_effect=RuntimeError('private storage detail')))
        app.analyze_project_files(files, generate_ai=True, project_metadata={'project_name': case})
    assert state.enhancement_status['ai']['status'] == expected
    assert state.report_data['findings'] == baseline.report_data['findings']
    assert state.report_pdf.startswith(b'%PDF') and state.report_markdown
    assert state.report_error is None
    assert state.patch_text == state.report_data['patch_text'] == ''
    assert 'private storage detail' not in state.report_markdown
    assert 'private diagnostic details' not in state.report_markdown
    if fault == 'knowledge_and_api_failure':
        assert state.enhancement_status['knowledge']['status'] == 'retrieval_failed'
    output = os.environ.get('CODEDATA_BATCH_D_ARTIFACT_DIR')
    if output:
        (Path(output) / case / ('fault-' + fault + '.json')).write_text(json.dumps({
            'fault': fault, 'enhancement_status': state.enhancement_status,
            'static_findings_unchanged': True, 'pdf_exported': True,
            'canonical_counts': state.report_data['conclusions']['counts'],
            'patch_exported': bool(state.patch_text)}, ensure_ascii=False, indent=2))
