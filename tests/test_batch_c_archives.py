"""ZIP safety and bounded local/GitHub parity without executing uploaded sources."""
from io import BytesIO
from pathlib import Path
import stat
import zipfile

import pytest

from file_loader import materialize_project_uploads
from github_loader import _scan_repository
from project_archive import ArchiveBudget, inspect_zip, safe_relative_path


class Upload:
    def __init__(self, name, content):
        self.name, self.content = name, content
    def getvalue(self):
        return self.content


def archive_bytes(entries):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in entries:
            z.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize('name', ['../out.py', '/out.py', 'C:/out.py', 'a/../out.py', 'a\\..\\out.py', 'a/./out.py', 'a//out.py', 'a.py:stream', '\x00x.py', 'NUL.csv', 'file.py.', 'a\nb.py'])
def test_unsafe_paths_are_rejected(name):
    assert safe_relative_path(name) is None


def test_local_zip_preserves_directories_and_csv_and_never_reads_secrets(monkeypatch):
    contents = archive_bytes([('src/model.py', 'raise RuntimeError("never execute")'),
                              ('data/raw.csv', 'x,y\n1,0\n2,1\n'),
                              ('README.md', '# demo'), ('.env', 'do not read'),
                              ('.git/config', 'do not read'), ('notes.bin', b'\x00x'), ('../bad.py', 'bad')])
    original = zipfile.ZipFile.open
    opened = []
    def guard(self, name, *args, **kwargs):
        path = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert '.env' not in path and '.git/' not in path
        opened.append(path)
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'open', guard)
    with materialize_project_uploads([Upload('project.zip', contents)]) as files:
        assert {name for name, _ in files} == {'project.zip!/src/model.py', 'project.zip!/data/raw.csv', 'project.zip!/README.md'}
        paths = [path for _, path in files]
        assert all(path.is_file() for path in paths)
        assert len(files.skipped_files) == 4
        assert not any(path.parent.parent.joinpath('bad.py').is_file() for path in paths)
    assert not any(path.exists() for path in paths)
    assert 'notes.bin' not in opened


def test_local_zip_and_github_zip_use_same_content_policy(tmp_path):
    payload = archive_bytes([('data/raw.csv', 'x\n1\n'), ('src/model.py', 'x=1'), ('Makefile', 'all:\n\tpython src/model.py'), ('requirements.txt', 'pandas>=2')])
    (tmp_path / 'project.zip').write_bytes(payload)
    github = _scan_repository(tmp_path)
    with materialize_project_uploads([Upload('project.zip', payload)]) as local:
        assert sorted(name for name, _ in local) == sorted(name for name, _ in github)


def test_duplicate_and_case_conflicting_members_are_all_skipped(tmp_path):
    contents = archive_bytes([('A.py', 'x=1'), ('a.py', 'x=2'), ('ok.csv', 'x\n1\n')])
    files, skipped = [], []
    inspect_zip(contents, 'p.zip', tmp_path, files, skipped)
    assert [name for name, _ in files] == ['p.zip!/ok.csv']
    assert len(skipped) == 2 and all('冲突' in item['reason'] for item in skipped)


def test_archive_symlink_binary_and_corruption_are_explicit(tmp_path):
    link = zipfile.ZipInfo('link.py')
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    payload = archive_bytes([(link, '../../x'), ('binary.py', b'\x00binary'), ('ok.py', 'x=1')])
    files, skipped = [], []
    inspect_zip(payload, 'p.zip', tmp_path, files, skipped)
    assert len(files) == 1 and len(skipped) == 2
    inspect_zip(b'not a zip', 'bad.zip', tmp_path, files, skipped)
    assert 'BadZipFile' in skipped[-1]['reason']


@pytest.mark.parametrize('limits,entries,reason', [
    ({'max_members': 1}, [('a.py', 'x=1'), ('b.py', 'x=2')], '成员'),
    ({'max_member_bytes': 2}, [('a.py', 'x=1')], '大小'),
    ({'max_total_bytes': 4}, [('a.py', 'x=1'), ('b.py', 'x=2')], '字节'),
    ({'max_files': 1}, [('a.py', 'x=1'), ('b.py', 'x=2')], '文件'),
    ({'max_ratio': 2}, [('a.csv', 'x\n' + '0\n' * 1000)], '压缩比'),
])
def test_archive_resource_limits_are_reported(tmp_path, limits, entries, reason):
    files, skipped = [], []
    inspect_zip(archive_bytes(entries), 'p.zip', tmp_path, files, skipped, ArchiveBudget(**limits))
    assert any(reason in item['reason'] for item in skipped)


def test_nested_archives_share_budget_and_depth(tmp_path):
    inner = archive_bytes([('a.py', 'x=1'), ('b.csv', 'x\n1')])
    outer = archive_bytes([('one.zip', inner), ('two.zip', inner)])
    files, skipped = [], []
    inspect_zip(outer, 'p.zip', tmp_path, files, skipped, ArchiveBudget(max_files=1))
    assert len(files) == 1 and skipped
    files, skipped = [], []
    inspect_zip(outer, 'p.zip', tmp_path, files, skipped, ArchiveBudget(max_depth=1))
    assert not files and all('深度' in item['reason'] for item in skipped)


def test_loose_upload_paths_and_private_files_are_checked_before_read():
    class Secret:
        name = '.env'
        def getvalue(self):
            pytest.fail('secret bytes must never be requested')
    with pytest.raises(ValueError, match='敏感'):
        with materialize_project_uploads([Secret()]):
            pass
    with materialize_project_uploads([Upload('a/x.py', b'x=1'), Upload('b/x.py', b'x=2')]) as files:
        assert [name for name, _ in files] == ['a/x.py', 'b/x.py']
