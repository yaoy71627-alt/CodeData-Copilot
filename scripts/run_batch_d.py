"""Offline four-case acceptance with pinned input hashes and reproducible evidence.

No .env loading, uploaded-code execution, live API requests or full prompt logging.
Use a fresh --output directory for each run. Original private cases stay local.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from regression_benchmark import summarize_cases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(root=ROOT):
    """Pin implementation, rules and dependency files; detect added/deleted sources."""
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith('.') or part.startswith('validation_') or part == '__pycache__' for part in relative.parts):
            continue
        if (path.suffix in ('.py', '.toml', '.ini') or path.name == 'requirements.txt'
                or ('knowledge' in relative.parts and path.suffix == '.md')
                or relative.as_posix() == 'tests/fixtures/batch_d/truth.json'):
            result[relative.as_posix()] = digest(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for case in ('titanic', 'cardekho'):
        for kind in ('notebook', 'csv'):
            parser.add_argument(f'--{case}-{kind}', type=Path)
    args = parser.parse_args(argv)
    truth_path = ROOT / 'tests/fixtures/batch_d/truth.json'
    truth = json.loads(truth_path.read_text(encoding='utf-8'))
    env = dict(os.environ, PYTHON_DOTENV_DISABLED='1')
    inputs = {}
    for case in ('titanic', 'cardekho'):
        for (alias, expected), kind in zip(truth['inputs'][case].items(), ('notebook', 'csv')):
            key = f'CODEDATA_{case.upper()}_{kind.upper()}'
            path = getattr(args, case + '_' + kind) or (Path(env[key]) if env.get(key) else None)
            if path is None or not path.is_file():
                parser.error(f'Missing original case: --{case}-{kind} or {key}')
            if digest(path) != expected:
                parser.error(f'{case}/{kind} hash differs from the annotated case; do not reuse this truth for a different input')
            env[key] = str(path.resolve())
            inputs[case + '/' + alias] = expected
    # Pin every public fixture including provenance and licences.
    for p in sorted((ROOT / 'tests/fixtures').rglob('*')):
        if p.is_file():
            inputs[p.relative_to(ROOT).as_posix()] = digest(p)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be a fresh or empty directory; prior evidence is never silently reused')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {'CODEDATA_BATCH_D_ARTIFACT_DIR': 'cases', 'CODEDATA_TEST_ARTIFACT_DIR': 'real_cases',
                        'CODEDATA_BATCH_B_ARTIFACT_DIR': 'batch_b_cases', 'CODEDATA_BATCH_C_ARTIFACT_DIR': 'pinned_cases'}.items():
        env[name] = str(output / value)
    source_hashes = source_snapshot()
    command = [sys.executable, '-m', 'pytest', '-q', '-m', 'not e2e', '-ra']
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                'platform': platform.platform(), 'git_commit': None,
                'version_note': 'Archive project has no Git commit; source-hashes.json pins the implementation.',
                'ruleset': truth['ruleset'], 'truth_sha256': digest(truth_path),
                'inputs': inputs, 'online_api': {'status': 'not_run', 'reason': 'Explicit local opt-in required; offline Mock results are not online evidence.'},
                'environment': {name: importlib.metadata.version(name) for name in
                    ('pytest','streamlit','pandas','scikit-learn','numpy','nbformat','openai','reportlab','pypdf','python-dotenv')},
                'command': [*command, '--junitxml=junit.xml']}
    (output / 'source-hashes.json').write_text(json.dumps(source_hashes, indent=2))
    with (output / 'collection.log').open('w') as log:
        collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    with (output / 'test-run.log').open('w') as log:
        completed = subprocess.run([*command, '--junitxml=' + str(output / 'junit.xml')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    results = {}
    for case in truth['cases']:
        path = output / 'cases' / case / 'benchmark.json'
        if path.is_file():
            results[case] = json.loads(path.read_text(encoding='utf-8'))
    if len(results) == 4:
        summary = summarize_cases(results)
        (output / 'benchmark-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    manifest.update({'finished_utc': datetime.now(timezone.utc).isoformat(), 'collection_exit_code': collected.returncode,
                     'pytest_exit_code': completed.returncode, 'evaluated_cases': sorted(results),
                     'source_unchanged': source_hashes == source_snapshot()})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({key: manifest[key] for key in ('pytest_exit_code', 'evaluated_cases', 'source_unchanged')}, ensure_ascii=False))
    return completed.returncode or collected.returncode or (0 if len(results) == 4 and manifest['source_unchanged'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
