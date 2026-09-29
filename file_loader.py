"""Streamlit 上传文件的校验、解码和临时保存工具。"""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


MAX_FILE_SIZE = 25 * 1024 * 1024
CODE_SUFFIXES = {".py", ".ipynb"}
DATA_SUFFIXES = {".csv"}
PROJECT_SUFFIXES = CODE_SUFFIXES | DATA_SUFFIXES


def validate_uploaded_file(uploaded_file, allowed_suffixes):
    """校验上传对象的文件名、扩展名和大小。"""
    if uploaded_file is None:
        raise ValueError("尚未选择文件")

    safe_name = Path(uploaded_file.name).name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in allowed_suffixes:
        expected = "、".join(sorted(allowed_suffixes))
        raise ValueError(f"不支持 {suffix or '无扩展名'} 文件，仅支持：{expected}")

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError(f"{safe_name} 是空文件")
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(f"{safe_name} 超过 25 MB 限制")

    return safe_name, suffix, content


def decode_python_file(content):
    """兼容常见编码读取 Python 源码。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Python 文件编码无法识别，请另存为 UTF-8 后重试")


@contextmanager
def materialize_uploads(code_file, data_file):
    """把两个上传对象安全写入一次性临时目录，退出后自动清理。"""
    code_name, code_suffix, code_content = validate_uploaded_file(
        code_file, CODE_SUFFIXES
    )
    data_name, _, data_content = validate_uploaded_file(data_file, DATA_SUFFIXES)

    with TemporaryDirectory(prefix="codedata_copilot_") as temp_dir:
        temp_path = Path(temp_dir)
        code_path = temp_path / code_name
        data_path = temp_path / data_name
        code_path.write_bytes(code_content)
        data_path.write_bytes(data_content)
        yield {
            "code_path": code_path,
            "code_suffix": code_suffix,
            "code_content": code_content,
            "data_path": data_path,
        }


@contextmanager
def materialize_project_uploads(uploaded_files):
    """Materialize a multi-file local project while preserving existing validation."""
    uploaded_files = list(uploaded_files or [])
    if not uploaded_files:
        raise ValueError("尚未选择项目文件")

    validated = []
    seen_names = set()
    for uploaded_file in uploaded_files:
        safe_name, _, content = validate_uploaded_file(uploaded_file, PROJECT_SUFFIXES)
        if safe_name.lower() in seen_names:
            raise ValueError(f"存在同名文件：{safe_name}，请重命名后重新上传")
        seen_names.add(safe_name.lower())
        validated.append((safe_name, content))

    with TemporaryDirectory(prefix="codedata_project_") as temp_dir:
        temp_path = Path(temp_dir)
        materialized = []
        for safe_name, content in validated:
            path = temp_path / safe_name
            path.write_bytes(content)
            materialized.append((safe_name, path))
        yield materialized
