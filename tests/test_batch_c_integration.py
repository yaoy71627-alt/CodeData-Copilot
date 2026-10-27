"""End-to-end offline multi-CSV/ZIP report and UI acceptance."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pandas as pd
from pypdf import PdfReader
import pytest
from streamlit.testing.v1 import AppTest

import app
from file_loader import materialize_project_uploads
from report import project_scope_lines


def analyze(files):
    state = SimpleNamespace()
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files}
    with patch.object(app.st, 'session_state', state):
        app.analyze_project_files(files, generate_ai=False)
    assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files} == before
    assert state.report_error is None and state.report_pdf.startswith(b'%PDF')
    return state


@pytest.fixture(scope='module')
def independent_state(tmp_path_factory):
    root = tmp_path_factory.mktemp('csv_association')
    raw = root / 'raw.csv'
    other = root / 'other.csv'
    pd.DataFrame({'age': [None if i < 8 else i for i in range(40)], 'target': [i % 2 for i in range(40)]}).to_csv(raw, index=False)
    pd.DataFrame({'unrelated': range(100), 'foreign_only': range(100)}).to_csv(other, index=False)
    code = root / 'model.py'
    code.write_text("import pandas as pd\nfrom sklearn.model_selection import train_test_split\ndf=pd.read_csv('raw.csv')\ndf=df.dropna()\nX=df[['age']]\ny=df['target']\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=3)\nmodel.fit(X_train,y_train)\nmodel.predict(X_test)")
    return root, [(other.name, other), (code.name, code), (raw.name, raw)], analyze([(other.name, other), (code.name, code), (raw.name, raw)])


def test_unrelated_first_csv_cannot_supply_model_fields_or_quantification(independent_state):
    _, _, state = independent_state
    reports = {e['file']: e['result'] for e in state.data_results}
    raw = reports['raw.csv']['duplicate_split_analysis']['modeling_data']
    unrelated = reports['other.csv']['duplicate_split_analysis']['modeling_data']
    assert raw['train_rows'] == 24 and raw['test_rows'] == 8
    assert raw['model_input_columns'] == ['age']
    assert unrelated['train_rows'] is None and unrelated['test_rows'] is None
    assert '未能关联' in unrelated['reason']
    assert reports['other.csv']['quantified_impacts'] == []
    assert {item['source_csv'] for item in state.quantified_impacts} == {'raw.csv'}
    assert state.analysis_facts['model_features'] == []
    assert state.analysis_facts['sample_definition']['unique_project_rows'] is None


def test_reordered_uploads_keep_findings_associations_and_scores(independent_state):
    _, files, state = independent_state
    reordered = analyze(list(reversed(files)))
    assert state.report_data['findings'] == reordered.report_data['findings']
    assert state.health['score'] == reordered.health['score']
    assert state.analysis_facts['dataset_graph'] == reordered.analysis_facts['dataset_graph']


def test_page_markdown_pdf_scope_is_identical_and_dataset_can_be_switched(independent_state):
    _, _, state = independent_state
    compact = lambda value: ''.join(value.split())
    text = compact('\n'.join(p.extract_text() for p in PdfReader(BytesIO(state.report_pdf)).pages))
    for line in project_scope_lines(state.analysis_facts):
        assert line in state.report_markdown
        assert compact(line) in text
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30)
    for key, value in vars(state).items():
        at.session_state[key] = value
    at.session_state['active_page'] = 'Data Quality'
    at.run()
    assert not at.exception
    at.selectbox[0].select('raw.csv').run()
    assert not at.exception
    assert any(metric.label == 'Rows' and metric.value == '40' for metric in at.metric)


def test_a_split_in_another_pipeline_must_not_remove_no_split_finding(tmp_path):
    one, two = tmp_path / 'one.py', tmp_path / 'two.py'
    one.write_text('from sklearn.model_selection import train_test_split\na,b=train_test_split(X,test_size=.3)')
    two.write_text('model.fit(X,y)')
    state = analyze([('one.py', one), ('two.py', two)])
    assert any(f['rule_id'] == 'CODE_NO_VALIDATION_SPLIT' and f['location']['file'] == 'two.py' for f in state.report_data['findings'])


@pytest.fixture(scope='module', params=['direct', 'wrapped_zip'])
def telco_state(request):
    base = Path(__file__).parent / 'fixtures'
    old, new = base / 'batch_b/telco', base / 'batch_c/telco'
    files = [(p.relative_to(old).as_posix(), p) for p in sorted(old.rglob('*.py'))]
    files += [('README.md', old / 'README.md'), ('Makefile', new / 'Makefile')]
    files += [(p.relative_to(new).as_posix(), p) for p in sorted(new.rglob('*.csv'))]
    if request.param == 'direct':
        state = analyze(files)
    else:
        payload = BytesIO()
        with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, path in files:
                archive.writestr('project/' + name, path.read_bytes())
            archive.writestr('project/.env', 'never-read-this-placeholder')
        upload = SimpleNamespace(name='telco.zip', getvalue=lambda: payload.getvalue())
        with materialize_project_uploads([upload]) as imported:
            state = analyze(imported)
    out = os.environ.get('CODEDATA_BATCH_C_ARTIFACT_DIR')
    if out:
        path = Path(out) / ('telco_' + request.param)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'report.pdf').write_bytes(state.report_pdf)
        (path / 'report.md').write_text(state.report_markdown)
        (path / 'facts.json').write_text(json.dumps(state.analysis_facts, ensure_ascii=False, indent=2, default=str))
    return request.param, state


def test_real_telco_raw_train_test_keep_their_roles_and_counts(telco_state):
    mode, state = telco_state
    reports = {Path(e['file']).name: e['result'] for e in state.data_results}
    assert {name: r['shape']['rows'] for name, r in reports.items()} == {'IBM-Telco-Customer-Churn.csv': 7043, 'train_df.csv': 4930, 'test_df.csv': 2113}
    graph = {Path(d['file']).name: d for d in state.analysis_facts['dataset_graph']['datasets']}
    assert graph['train_df.csv']['roles'] == ['train'] and graph['test_df.csv']['roles'] == ['test']
    assert Path(graph['train_df.csv']['parents'][0]).name == 'IBM-Telco-Customer-Churn.csv'
    assert Path(graph['test_df.csv']['parents'][0]).name == 'IBM-Telco-Customer-Churn.csv'
    assert state.analysis_facts['sample_definition']['unique_project_rows'] is None
    assert state.health['dataset_scoring']['group_count'] == 1
    assert state.analysis_facts['model_features'] == []
    assert not any(f['rule_id'] == 'CODE_NO_CROSS_VALIDATION' for f in state.report_data['findings'])
    assert not any(f['rule_id'] == 'DQ_SCHEMA_DIFFERENCE' for f in state.report_data['findings'])
    if mode == 'wrapped_zip':
        assert state.analysis_facts['analysis_scope']['coverage_complete'] is False
        assert any('.env' in s['path'] for s in state.analysis_facts['analysis_scope']['skipped_files'])


def test_telco_export_and_canonical_scope_are_readable(telco_state):
    _, state = telco_state
    compact = lambda value: ''.join(value.split())
    text = compact('\n'.join(p.extract_text() for p in PdfReader(BytesIO(state.report_pdf)).pages))
    for line in project_scope_lines(state.analysis_facts):
        assert line in state.report_markdown
        assert compact(line) in text
    assert state.patch_text == state.report_data['patch_text']
    assert '各 CSV 行数是文件内记录数' in state.report_markdown
