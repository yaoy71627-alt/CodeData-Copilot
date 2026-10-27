"""Bounded, shared ZIP inspection. Member code is never imported or executed."""
from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path, PurePosixPath
import stat
import unicodedata
import zipfile

from static_review_rules import is_dependency_manifest


SUPPORTED_SUFFIXES = {".py", ".ipynb", ".csv", ".md"}
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules", "__macosx"}


def safe_relative_path(name):
    value = unicodedata.normalize("NFC", str(name).replace("\\", "/"))
    parts = value.split("/")
    if not value or value.startswith('/') or any(p in {"", ".", ".."} or ':' in p for p in parts) or '\x00' in value:
        return None
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if any(p != p.rstrip(' .') or p.split('.')[0].upper() in reserved or any(ord(c) < 32 for c in p) for p in parts):
        return None
    return PurePosixPath(*parts).as_posix()


def private_path(name):
    parts = str(name).replace("\\", "/").lower().split('/')
    return any(p in IGNORED_PARTS or p.startswith('.env') for p in parts)


def supported_path(name):
    return (PurePosixPath(name).suffix.lower() in SUPPORTED_SUFFIXES
            or is_dependency_manifest(name) or PurePosixPath(name).name.lower() == 'makefile')


@dataclass
class ArchiveBudget:
    max_members: int = 1000
    max_files: int = 100
    max_member_bytes: int = 25 * 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    max_archive_bytes: int = 25 * 1024 * 1024
    max_ratio: int = 100
    max_depth: int = 2
    members: int = 0
    bytes_read: int = 0
    files: int = 0
    paths: set = field(default_factory=set)


class ProjectFiles(list):
    def __init__(self, files=(), skipped_files=()):
        super().__init__(files)
        self.skipped_files = list(skipped_files)


def inspect_zip(source, logical_name, extraction_root, files, skipped, budget=None, *, depth=1):
    """One budget across nested archives; never extractall, follow links or overwrite."""
    budget = budget or ArchiveBudget()

    def skip(path, reason, size=None):
        skipped.append({"path": path, "reason": reason, "size_mb": round(size / 1048576, 2) if size is not None else None})

    if depth > budget.max_depth:
        skip(logical_name, "嵌套 ZIP 深度超过限制")
        return
    try:
        size = len(source) if isinstance(source, bytes) else Path(source).stat().st_size
        if size > budget.max_archive_bytes:
            skip(logical_name, "ZIP 压缩大小超过限制", size)
            return
        with zipfile.ZipFile(io.BytesIO(source) if isinstance(source, bytes) else source) as archive:
            members = archive.infolist()
            if budget.members + len(members) > budget.max_members:
                skip(logical_name, "ZIP 成员总数超过共享预算")
                return
            budget.members += len(members)
            normalized = [safe_relative_path(i.filename.rstrip('/')) for i in members]
            counts = {}
            for name in normalized:
                if name:
                    counts[name.casefold()] = counts.get(name.casefold(), 0) + 1
            for info, name in sorted(zip(members, normalized), key=lambda pair: pair[0].filename):
                if info.is_dir():
                    continue
                logical = f"{logical_name}!/{info.filename}"
                if not name:
                    skip(logical, "unsafe archive member path：拒绝绝对路径、路径遍历或非法目录")
                    continue
                if private_path(name):
                    skip(logical, "敏感配置或忽略目录：仅检查名称，未读取内容")
                    continue
                mode = (info.external_attr >> 16) & 0xffff
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
                    skip(logical, "ZIP 符号链接或特殊文件未读取")
                    continue
                if counts[name.casefold()] > 1:
                    skip(logical, "ZIP 存在重复或大小写冲突路径，未选择任意副本")
                    continue
                if info.flag_bits & 1:
                    skip(logical, "加密 ZIP 成员无法检查")
                    continue
                nested = PurePosixPath(name).suffix.lower() == '.zip'
                if not nested and not supported_path(name):
                    skip(logical, "不支持的文件类型，未读取内容", info.file_size)
                    continue
                if budget.files >= budget.max_files or len(files) >= budget.max_files:
                    skip(logical, "分析文件总数超过共享预算")
                    continue
                if info.file_size > budget.max_member_bytes or info.file_size / max(info.compress_size, 1) > budget.max_ratio:
                    skip(logical, "ZIP 成员大小或压缩比超过限制", info.file_size)
                    continue
                if budget.bytes_read + info.file_size > budget.max_total_bytes:
                    skip(logical, "ZIP 解压字节数超过共享预算", info.file_size)
                    continue
                budget.bytes_read += info.file_size
                try:
                    with archive.open(info) as handle:
                        data = handle.read(min(info.file_size + 1, budget.max_member_bytes + 1))
                    if len(data) != info.file_size:
                        skip(logical, "ZIP 成员实际大小与元数据不一致")
                        continue
                except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, NotImplementedError) as exc:
                    skip(logical, f"ZIP 成员无法安全读取（{type(exc).__name__}）")
                    continue
                if nested:
                    inspect_zip(data, logical, extraction_root, files, skipped, budget, depth=depth + 1)
                    continue
                if b'\x00' in data[:8192]:
                    skip(logical, "文件扩展名为文本但内容疑似二进制")
                    continue
                key = logical.casefold()
                if key in budget.paths:
                    skip(logical, "导入路径冲突，未覆盖已有文件")
                    continue
                budget.paths.add(key)
                folder = hashlib.sha256(logical_name.encode()).hexdigest()[:16]
                destination = Path(extraction_root) / folder / Path(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                files.append((logical, destination))
                budget.files += 1
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        skip(logical_name, f"ZIP 无法安全检查（{type(exc).__name__}）")
