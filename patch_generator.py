"""Generate reviewable unified diffs without modifying analyzed projects."""

from difflib import unified_diff
from pathlib import PurePosixPath
import re


def _safe_patch_path(location):
    file_name = str(location or "suggested_fix.py").split(" · ", 1)[0].replace("\\", "/")
    parts = [part for part in PurePosixPath(file_name).parts if part not in {"", ".", "..", "/"}]
    return "/".join(parts) or "suggested_fix.py"


def _suggestion_diff(suggestion):
    before = suggestion.get("before_code") or suggestion.get("original_code")
    after = suggestion.get("after_code") or suggestion.get("corrected_code")
    if not before or not after:
        return ""
    path = _safe_patch_path(suggestion.get("location"))
    return "\n".join(unified_diff(
        str(before).splitlines(),
        str(after).splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))


def build_unified_patch(suggestions):
    """Return one standard unified-diff document for suggestions with real source."""
    patches = []
    seen = set()
    for suggestion in suggestions or []:
        patch = _suggestion_diff(suggestion)
        normalized = re.sub(r"^@@.*@@$", "@@", patch, flags=re.MULTILINE)
        if patch and normalized not in seen:
            patches.append(patch)
            seen.add(normalized)
    return "\n\n".join(patches) + ("\n" if patches else "")
