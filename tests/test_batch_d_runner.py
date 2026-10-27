"""The offline runner refuses missing or mismatched original cases before pytest."""
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).parents[1] / 'scripts/run_batch_d.py'


def test_runner_rejects_missing_cases_without_creating_evidence(tmp_path):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith('CODEDATA_TITANIC')}
    out = tmp_path / 'evidence'
    result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(out)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Missing original case' in result.stderr
    assert not out.exists()


def test_runner_rejects_changed_input_before_claiming_acceptance(tmp_path):
    source = tmp_path / 'changed.ipynb'
    source.write_text('{}')
    out = tmp_path / 'evidence'
    result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(out), '--titanic-notebook', str(source)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'hash differs' in result.stderr
    assert not out.exists()


def test_source_snapshot_detects_added_deleted_changed_rules_and_dependencies(tmp_path):
    from scripts.run_batch_d import source_snapshot
    (tmp_path / 'model.py').write_text('value = 1')
    initial = source_snapshot(tmp_path)
    (tmp_path / 'new.py').write_text('value = 2')
    assert source_snapshot(tmp_path) != initial
    (tmp_path / 'new.py').unlink()
    assert source_snapshot(tmp_path) == initial
    (tmp_path / 'requirements.txt').write_text('pandas>=2')
    (tmp_path / 'knowledge').mkdir()
    (tmp_path / 'knowledge/rule.md').write_text('Review the evidence')
    pinned = source_snapshot(tmp_path)
    assert {'requirements.txt', 'knowledge/rule.md'} <= pinned.keys()
    (tmp_path / 'knowledge/rule.md').write_text('Changed rule')
    assert source_snapshot(tmp_path) != pinned
    (tmp_path / 'model.py').unlink()
    assert 'model.py' not in source_snapshot(tmp_path)
