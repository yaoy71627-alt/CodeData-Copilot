"""Safely import supported files from a public GitHub repository."""

from contextlib import contextmanager
import os
from pathlib import Path
import re
from subprocess import TimeoutExpired, run
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


SUPPORTED_SUFFIXES = {".py", ".ipynb", ".csv", ".md"}
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PROJECT_FILES = 100
CLONE_TIMEOUT_SECONDS = 300
IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "models", "checkpoints"
}
BINARY_SUFFIXES = {
    ".7z", ".avi", ".bin", ".bmp", ".ckpt", ".dll", ".dylib", ".exe", ".gif",
    ".gz", ".h5", ".hdf5", ".ico", ".jpeg", ".jpg", ".joblib", ".mp3", ".mp4",
    ".onnx", ".parquet", ".pdf", ".pkl", ".pickle", ".png", ".pt", ".pth", ".so",
    ".tar", ".tflite", ".webp", ".weights", ".xls", ".xlsx", ".zip",
}


class GitHubImportError(ValueError):
    """A user-facing import error with a stable category for the UI."""

    def __init__(self, category, message, skipped_files=None):
        super().__init__(message)
        self.category = category
        self.skipped_files = list(skipped_files or [])


class GitHubProjectFiles(list):
    """List-compatible import result with details about files not analyzed."""

    def __init__(self, files=(), skipped_files=()):
        super().__init__(files)
        self.skipped_files = list(skipped_files)


def validate_github_url(url):
    """Accept only a GitHub repository root, never arbitrary clone URLs."""
    if not isinstance(url, str):
        raise GitHubImportError("invalid_repository", "GitHub 仓库地址无效，请输入完整 URL。")
    parsed = urlsplit(url.strip())
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "https" or parsed.netloc.lower() != "github.com"
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or len(parts) != 2):
        raise GitHubImportError(
            "invalid_repository",
            "GitHub 仓库地址无效。请输入 https://github.com/owner/repository 格式的公开仓库地址。",
        )
    owner, repo = parts
    repo = repo.removesuffix(".git")
    if (not owner or not repo
            or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in (owner, repo))):
        raise GitHubImportError("invalid_repository", "GitHub 仓库地址包含无效的用户名或仓库名。")
    if owner in {".", ".."} or repo in {".", ".."}:
        raise GitHubImportError("invalid_repository", "GitHub 仓库地址无效。")
    return f"https://github.com/{owner}/{repo}.git"


def _git_output(stdout="", stderr=""):
    """Format captured Git output so Streamlit can show the real failure."""
    def as_text(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value or "")

    blocks = []
    stderr_text = as_text(stderr).strip()
    stdout_text = as_text(stdout).strip()
    if stderr_text:
        blocks.append(f"Git stderr:\n{stderr_text}")
    if stdout_text:
        blocks.append(f"Git stdout:\n{stdout_text}")
    return "\n\n".join(blocks) or "Git 没有返回 stdout 或 stderr。"


def _clone_error(stderr, stdout=""):
    """Classify a Git failure without hiding its original stdout/stderr."""
    lowered = f"{stderr or ''}\n{stdout or ''}".lower()
    git_output = _git_output(stdout, stderr)
    invalid_markers = (
        "repository not found", "not found", "authentication failed", "could not read username",
        "does not appear to be a git repository", "access denied", "permission denied",
    )
    network_markers = (
        "could not resolve host", "failed to connect", "connection timed out",
        "couldn't connect", "connection refused", "unable to access",
        "network is unreachable", "connection reset", "proxy", "ssl", "tls",
        "rpc failed", "gnutls", "recv failure", "early eof",
    )
    if any(marker in lowered for marker in invalid_markers):
        return GitHubImportError(
            "invalid_repository",
            "无法访问该仓库：仓库不存在、不是公开仓库，或当前账号没有访问权限。"
            f"\n\n{git_output}",
        )
    if any(marker in lowered for marker in network_markers):
        return GitHubImportError(
            "network_problem",
            "GitHub 网络连接失败。以下是 Git 的原始输出："
            f"\n\n{git_output}",
        )
    return GitHubImportError(
        "clone_failed",
        f"GitHub 仓库克隆失败。以下是 Git 的原始输出：\n\n{git_output}",
    )


def _looks_binary(path):
    """Detect a binary payload even when it has a supported text extension."""
    try:
        with path.open("rb") as source:
            return b"\x00" in source.read(8192)
    except OSError:
        return False


def _scan_repository(root):
    files = []
    skipped = []
    supported_text = "、".join(sorted(SUPPORTED_SUFFIXES))

    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories = []
        for directory_name in sorted(directory_names):
            directory_path = current_path / directory_name
            relative = directory_path.relative_to(root).as_posix()
            if directory_name.lower() in IGNORED_DIRECTORIES:
                skipped.append({"path": relative + "/", "reason": "已忽略目录", "size_mb": None})
            elif directory_path.is_symlink():
                skipped.append({"path": relative + "/", "reason": "符号链接目录未扫描", "size_mb": None})
            else:
                kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in sorted(file_names):
            path = current_path / file_name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                skipped.append({"path": relative, "reason": "符号链接文件未分析", "size_mb": None})
                continue
            try:
                size = path.stat().st_size
            except OSError as exc:
                skipped.append({"path": relative, "reason": f"无法读取文件信息：{exc}", "size_mb": None})
                continue
            size_mb = round(size / (1024 * 1024), 2)
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                reason = ("二进制文件，不参与分析" if suffix in BINARY_SUFFIXES
                          else f"不支持的文件类型（仅分析 {supported_text}）")
                skipped.append({"path": relative, "reason": reason, "size_mb": size_mb})
                continue
            if size > MAX_FILE_SIZE:
                skipped.append({
                    "path": relative,
                    "reason": f"文件超过 {MAX_FILE_SIZE // (1024 * 1024)} MB 上限",
                    "size_mb": size_mb,
                })
                continue
            if _looks_binary(path):
                skipped.append({"path": relative, "reason": "检测到二进制内容", "size_mb": size_mb})
                continue
            if len(files) >= MAX_PROJECT_FILES:
                skipped.append({
                    "path": relative,
                    "reason": f"已达到最多 {MAX_PROJECT_FILES} 个分析文件的限制",
                    "size_mb": size_mb,
                })
                continue
            files.append((relative, path))

    return GitHubProjectFiles(files, skipped)


@contextmanager
def load_github_project(url):
    """Yield supported files and skip details while the temporary clone exists."""
    clone_url = validate_github_url(url)
    with TemporaryDirectory(prefix="codedata_github_") as directory:
        temporary_root = Path(directory).resolve()
        if not temporary_root.is_dir():
            raise GitHubImportError(
                "temporary_directory_error",
                f"无法创建临时克隆目录：{temporary_root}",
            )
        root = temporary_root / "repository"
        if root.exists():
            raise GitHubImportError(
                "temporary_directory_error",
                f"临时克隆目标已存在，无法安全继续：{root}",
            )
        try:
            result = run(
                ["git", "clone", "--depth", "1", "--single-branch", clone_url, str(root)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CLONE_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitHubImportError("git_missing", "未找到 Git。请安装 Git 并确认 git 命令可用后重试。") from exc
        except TimeoutExpired as exc:
            raise GitHubImportError(
                "clone_timeout",
                f"GitHub 仓库克隆超过 {CLONE_TIMEOUT_SECONDS} 秒，已停止。"
                "以下是超时前捕获的 Git 输出："
                f"\n\n{_git_output(getattr(exc, 'stdout', ''), getattr(exc, 'stderr', ''))}",
            ) from exc
        except OSError as exc:
            raise GitHubImportError("network_problem", f"启动 Git 或访问网络时失败：{exc}") from exc
        if result.returncode:
            raise _clone_error(
                getattr(result, "stderr", ""),
                getattr(result, "stdout", ""),
            )
        if not root.is_dir():
            raise GitHubImportError(
                "temporary_directory_error",
                "Git 返回成功，但临时仓库目录不存在。"
                f"\n\n{_git_output(getattr(result, 'stdout', ''), getattr(result, 'stderr', ''))}",
            )

        project_files = _scan_repository(root)
        if not project_files:
            detail = ""
            if project_files.skipped_files:
                examples = [f"{item['path']}（{item['reason']}）"
                            for item in project_files.skipped_files[:3]]
                detail = f" 已跳过：{'；'.join(examples)}。"
            raise GitHubImportError(
                "no_supported_files",
                "仓库中没有可分析的 .py、.ipynb、.csv 或 .md 文件"
                f"（单文件上限 {MAX_FILE_SIZE // (1024 * 1024)} MB）。{detail}",
                project_files.skipped_files,
            )
        yield project_files
