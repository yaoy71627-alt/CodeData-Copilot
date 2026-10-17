"""Export full Python-file patches verified against exact in-memory source snapshots."""

import ast
from difflib import unified_diff
from pathlib import PurePosixPath
import re


def _safe_patch_path(location):
    path = str(location or "").replace("\\", "/")
    parts = PurePosixPath(path).parts
    if not path or path.startswith("/") or re.match(r"^[A-Za-z]:", path) or any(part == ".." for part in parts):
        return None
    if any(char in path for char in "\r\n\t\0"):
        return None
    if not path.endswith(".py") or any(char.isspace() for char in path):
        return None
    return path


def _file_diff(before, after, path):
    lines = unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                         fromfile="a/" + path, tofile="b/" + path)
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines)


def apply_verified_diff(source, patch):
    """Strict single-file unified-diff application in memory; no fuzzy matching or IO."""
    lines = patch.splitlines(keepends=True)
    if len(lines) < 3 or not lines[0].startswith("--- a/") or not lines[1].startswith("+++ b/"):
        raise ValueError("缺少完整的单文件 diff 头。")
    original = source.splitlines(keepends=True)
    output, cursor, index = [], 0, 2
    while index < len(lines):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?", lines[index])
        if not match:
            raise ValueError("不支持的 diff hunk。")
        old_start, old_count = int(match[1]), int(match[2] or 1)
        new_start, new_count = int(match[3]), int(match[4] or 1)
        start = old_start - 1 if old_count else old_start
        if start < cursor or start > len(original):
            raise ValueError("diff 位置越界或重叠。")
        output.extend(original[cursor:start]); cursor = start
        expected_new = new_start - 1 if new_count else new_start
        if len(output) != expected_new:
            raise ValueError("diff 新文件行号不匹配。")
        index += 1
        removed, added = 0, 0
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if not line or line[0] not in " +-":
                raise ValueError("无效的 diff 内容。")
            tag, content = line[0], line[1:]
            if index + 1 < len(lines) and lines[index + 1].startswith("\\ No newline at end of file"):
                content = content[:-1] if content.endswith("\n") else content
                index += 1
            if tag in " -":
                if cursor >= len(original) or original[cursor] != content:
                    raise ValueError("diff 原始行与源快照不一致。")
                cursor += 1; removed += 1
            if tag in " +":
                output.append(content); added += 1
            index += 1
        if removed != old_count or added != new_count:
            raise ValueError("diff hunk 行数不匹配。")
    output.extend(original[cursor:])
    return "".join(output)


def build_unified_patch(suggestions):
    """Only verified full-file snapshots may be exported; never write analyzed files."""
    grouped = {}
    for item in suggestions or []:
        if item.get("validation_status") not in {"Statically Validated", "Runtime Verified"}:
            continue
        proof = item.get("patch_validation") or {}
        path = _safe_patch_path(item.get("file"))
        before = item.get("before_code") or item.get("original_code")
        after = item.get("after_code") or item.get("corrected_code")
        if (not path or not before or not after or not item.get("executable")
                or proof.get("scope") != "python_file" or not proof.get("eligible")
                or not proof.get("project_applicable") or proof.get("cell") is not None
                or proof.get("file") != item.get("file") or proof.get("line") != item.get("line")):
            continue
        source, target = proof.get("before_source"), proof.get("after_source")
        start, end = proof.get("start"), proof.get("end")
        if (not isinstance(source, str) or not isinstance(target, str)
                or not isinstance(start, int) or not isinstance(end, int)
                or not (0 <= start < end <= len(source)) or source[start:end] != str(before)
                or proof.get("before_code") != str(before) or proof.get("after_code") != str(after)
                or source[:start] + str(after) + source[end:] != target
                or source[:start].count("\n") + 1 != item.get("line")):
            continue
        grouped.setdefault(path, []).append((source, start, end, str(after)))
    patches = []
    for path, entries in sorted(grouped.items()):
        # All edits must agree on one source version. No overlapping proposals
        # or multiple independent diffs for the same file are exported.
        source = entries[0][0]
        if any(entry[0] != source for entry in entries):
            continue
        edits = sorted(set((start, end, after) for _, start, end, after in entries))
        if any(left[1] > right[0] for left, right in zip(edits, edits[1:])):
            continue
        target = source
        for start, end, after in reversed(edits):
            target = target[:start] + after + target[end:]
        try:
            ast.parse(target)
            patch = _file_diff(source, target, path)
            if not patch or apply_verified_diff(source, patch) != target:
                continue
        except (SyntaxError, ValueError):
            continue
        patches.append(patch)
    return "".join(patches)
