"""CSV association, ambiguity, scopes and source-group score regression."""
from pathlib import Path

import pytest

from data_flow_analyzer import analyze_python_data_flow, reconstruct_modeling_data
from project_lineage import build_dataset_graph, bind_dataset_flow, resolve_csv, scoped_dataset_consistency
from project_score import calculate_project_score


def unit(code, name='src/model.py'):
    return {'file': name, 'code': code}


@pytest.mark.parametrize('path,code,names,expected', [
    ('data/a.csv', 'src/m.py', ['data/a.csv', 'other/a.csv'], 'data/a.csv'),
    ('../data/a.csv', 'src/m.py', ['data/a.csv'], 'data/a.csv'),
    ('a.csv', 'src/m.py', ['data/a.csv', 'other/a.csv'], None),
    ('a.csv', 'p.zip!/src/m.py', ['p.zip!/data/a.csv', 'q.zip!/data/a.csv'], 'p.zip!/data/a.csv'),
    ('a.csv', 'p.zip!/src/m.py', ['q.zip!/data/a.csv'], None),
    ('../../a.csv', 'p.zip!/src/m.py', ['a.csv'], None),
    ('other/train.csv', 'src/m.py', ['raw/train.csv'], None),
    ('data/train.csv', 'p.zip!/project/src/m.py', ['p.zip!/project/data/train.csv'], 'p.zip!/project/data/train.csv'),
    (None, 'src/m.py', ['a.csv'], None),
])
def test_paths_match_within_project_namespace_without_order_guessing(path, code, names, expected):
    assert resolve_csv(path, code, names)[0] == expected


def test_graph_tracks_raw_transform_split_artifacts_and_consumers():
    units = [unit("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nraw=pd.read_csv('data/raw.csv')\nclean=raw.drop(columns=['id'])\ntrain,test=train_test_split(clean,test_size=.3)\ntrain.to_csv('data/train.csv')\ntest.to_csv('data/test.csv')", 'src/pre.py'),
             unit("import pandas as pd\ntrain=pd.read_csv('data/train.csv')\ntest=pd.read_csv('data/test.csv')")]
    files = [(name, Path(name)) for name in ['data/raw.csv', 'data/train.csv', 'data/test.csv']]
    graph = build_dataset_graph(units, files)
    datasets = {d['file']: d for d in graph['datasets']}
    assert datasets['data/train.csv']['parents'] == ['data/raw.csv']
    assert datasets['data/test.csv']['roles'] == ['test']
    assert len(datasets['data/train.csv']['loaders']) == 1
    assert graph['transformations'] and all(d['association_status'] == 'matched' for d in datasets.values())


def test_explicit_cli_parameters_bound_without_running_make_or_docopt():
    code = "import pandas as pd\nfrom sklearn.model_selection import train_test_split\ndef main(input,out_dir):\n raw=pd.read_csv(input)\n train,test=train_test_split(raw,test_size=.3)\n train.to_csv(out_dir+'train.csv')\nmain(opt['--input'],opt['--out_dir'])"
    graph = build_dataset_graph([unit(code)], [('raw.csv', Path('raw.csv')), ('out/train.csv', Path('out/train.csv'))],
                                [{'file': 'Makefile', 'excerpt': '\tpython src/model.py --input=raw.csv --out_dir=out/'}])
    read, = graph['reads']
    assert read['dataset'] == 'raw.csv' and read['binding']['file'] == 'Makefile'
    assert graph['datasets'][0]['parents'] == ['raw.csv']


def test_cli_bindings_do_not_cross_independent_archives():
    source = "import pandas as pd\ndef main(input):\n df=pd.read_csv(input)\nmain(opt['--input'])"
    graph = build_dataset_graph([unit(source, 'b.zip!/src/model.py')], [('b.zip!/x.csv', Path('x.csv'))],
                                [{'file': 'a.zip!/Makefile', 'excerpt': 'python src/model.py --input=x.csv'}])
    assert graph['reads'][0]['dataset'] is None


def test_dynamic_or_multiple_sources_never_borrow_first_csv(tmp_path):
    file = tmp_path / 'a.csv'
    file.write_text('age,target\n10,0\n20,1\n')
    for source in ["import pandas as pd\ndf=pd.read_csv(unknown)",
                   "import pandas as pd\na=pd.read_csv('a.csv')\nb=pd.read_csv('b.csv')"]:
        code = source + "\nX=df\na,b=train_test_split(X,test_size=.5)"
        graph = build_dataset_graph([unit(code)], [('a.csv', file), ('b.csv', Path('b.csv'))])
        flow = bind_dataset_flow(graph, 'a.csv', file, [analyze_python_data_flow(code, file_name='src/model.py')])
        result = reconstruct_modeling_data(flow, file)
        assert flow['association_status'] == 'unassociated'
        assert result['train_rows'] is None and result['test_rows'] is None
        assert '未能关联' in result['reason']


def test_missing_relative_source_does_not_rebind_to_an_existing_same_basename(tmp_path):
    code = "import pandas as pd\na=pd.read_csv('raw/train.csv')\nb=pd.read_csv('other/train.csv')"
    path = tmp_path / 'train.csv'
    path.write_text('x\n1\n')
    graph = build_dataset_graph([unit(code)], [('raw/train.csv', path)])
    assert graph['reads'][0]['dataset'] == 'raw/train.csv'
    assert graph['reads'][1]['dataset'] is None
    flow = bind_dataset_flow(graph, 'raw/train.csv', path, [analyze_python_data_flow(code, file_name='src/model.py')])
    assert flow['association_status'] == 'unassociated'


def test_legacy_replay_must_not_rank_a_known_unrelated_csv_flow(tmp_path):
    path = tmp_path / 'other.csv'
    path.write_text('x,y\n1,0\n2,1\n')
    flow = analyze_python_data_flow("import pandas as pd\nfrom sklearn.model_selection import train_test_split\ndf=pd.read_csv('actual.csv')\nX=df[['x']]\ny=df['y']\na,b,c,d=train_test_split(X,y,test_size=.5)", file_name='model.py')
    result = reconstruct_modeling_data(flow, path)
    assert result['test_rows'] is None and '未能关联' in result['reason']


def test_dynamic_reader_is_not_the_legacy_implicit_dataframe_api(tmp_path):
    path = tmp_path / 'unknown.csv'
    path.write_text('x,y\n1,0\n2,1\n')
    flow = analyze_python_data_flow("import pandas as pd\nfrom sklearn.model_selection import train_test_split\ndf=pd.read_csv(path)\nX=df[['x']]\ny=df['y']\na,b,c,d=train_test_split(X,y,test_size=.5)")
    result = reconstruct_modeling_data(flow, path)
    assert result['test_rows'] is None and '未能关联' in result['reason']


def test_bound_file_rejects_replay_for_another_dataset(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('x\n1\n2\n'); b.write_text('z\n1\n')
    code = "import pandas as pd\ndf=pd.read_csv('a.csv')"
    graph = build_dataset_graph([unit(code)], [('a.csv', a), ('b.csv', b)])
    flow = bind_dataset_flow(graph, 'a.csv', a, [analyze_python_data_flow(code, file_name='src/model.py')])
    assert reconstruct_modeling_data(flow, b)['test_rows'] is None
    assert reconstruct_modeling_data(flow, b)['status'] == 'Unable to reconstruct'


@pytest.mark.parametrize('options', [',nrows=5', ',skiprows=1', ',index_col=0', ",usecols=['x']", ',header=None'])
def test_unverified_reader_options_cannot_produce_exact_model_row_counts(tmp_path, options):
    file = tmp_path / 'a.csv'
    file.write_text('x\n1\n2\n3\n')
    code = "import pandas as pd\ndf=pd.read_csv('a.csv'" + options + ")\n"
    graph = build_dataset_graph([unit(code)], [('a.csv', file)])
    assert graph['reads'][0]['dataset'] == 'a.csv'
    flow = bind_dataset_flow(graph, 'a.csv', file, [analyze_python_data_flow(code, file_name='src/model.py')])
    assert flow['association_status'] == 'unassociated'
    assert reconstruct_modeling_data(flow, file)['test_rows'] is None


def test_two_independent_model_flows_are_not_ranked_to_guess_one(tmp_path):
    a = tmp_path / 'a.csv'; a.write_text('x\n1\n')
    units = [unit("import pandas as pd\ndf=pd.read_csv('a.csv')\nmodel.fit(df)", name) for name in ['one.py', 'two.py']]
    units[1]['code'] += '\nmodel.predict(df)'
    graph = build_dataset_graph(units, [('a.csv', a)])
    flow = bind_dataset_flow(graph, 'a.csv', a, [analyze_python_data_flow(u['code'], file_name=u['file']) for u in units])
    assert flow['association_status'] == 'unassociated'
    assert '多个' in flow['association_reason']


def test_pandas_parameter_shadowing_is_not_treated_as_verified_reader():
    code = "import pandas as pd\ndef custom(pd):\n data=pd.read_csv('raw.csv')\ncustom(unknown)"
    assert not build_dataset_graph([unit(code)], [('raw.csv', Path('raw.csv'))])['reads']


def test_schema_comparison_excludes_raw_derived_and_unrelated_files():
    entries = [{'file': name, 'result': {'data_types': types}} for name, types in [
        ('raw.csv', {'id': 'int64', 'x': 'int64'}), ('train.csv', {'x': 'float64'}),
        ('test.csv', {'x': 'object'}), ('other.csv', {'independent': 'int64'})]]
    graph = {'datasets': [{'file': e['file'], 'parents': ['raw.csv'] if e['file'] in {'train.csv', 'test.csv'} else [], 'loaders': []} for e in entries]}
    issues = scoped_dataset_consistency(entries, graph)
    assert len(issues) == 1 and issues[0]['files'] == ['test.csv', 'train.csv']
    assert issues[0]['verification_status'] == 'needs_review'
    assert len(graph['schema_comparison_scope']) == 6


def test_source_group_score_cannot_be_boosted_by_clean_derived_files():
    bad = {'file': 'raw.csv', 'result': {'shape': {'rows': 100}, 'issues': [{'rule_id': 'BAD', 'category': 'Missing', 'severity': 'High'}]}}
    good = {'file': 'train.csv', 'result': {'shape': {'rows': 80}, 'issues': []}}
    graph = {'datasets': [{'file': 'raw.csv', 'parents': []}, {'file': 'train.csv', 'parents': ['raw.csv']}]}
    baseline = calculate_project_score([], [bad], dataset_graph=graph)
    combined = calculate_project_score([], [bad, good], dataset_graph=graph)
    assert baseline['breakdown']['数据质量'] == combined['breakdown']['数据质量']
    assert combined['dataset_scoring']['group_count'] == 1
