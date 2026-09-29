"""Project-level duplicate artifact detection and issue consolidation."""

from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import io
from pathlib import Path
import re
import tokenize

import nbformat


NEAR_DUPLICATE_THRESHOLD = 0.95
ISSUE_SNIPPET_THRESHOLD = 0.85


def _decode_python(path):
    content = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _standardize_lines(text):
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _notebook_code(path):
    with open(path, "r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    return "\n\n".join(
        _standardize_lines(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def _remove_python_comments(text):
    """Remove comments without treating # inside a string as a comment."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        kept = [token for token in tokens if token.type not in {tokenize.COMMENT, tokenize.ENCODING}]
        return tokenize.untokenize(kept)
    except (tokenize.TokenError, IndentationError):
        return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _near_normalize(text):
    without_comments = _remove_python_comments(_standardize_lines(text))
    return re.sub(r"\s+", "", without_comments)


def _artifact_content(name, path):
    suffix = Path(name).suffix.casefold()
    if suffix == ".ipynb":
        code = _notebook_code(path)
    elif suffix == ".py":
        code = _decode_python(path)
    else:
        return None
    exact = _standardize_lines(code)
    return {
        "file": name,
        "suffix": suffix,
        "exact_text": exact,
        "exact_hash": hashlib.sha256(exact.encode("utf-8")).hexdigest(),
        "near_text": _near_normalize(code),
    }


def _union(parent, left, right):
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    root_left, root_right = find(left), find(right)
    if root_left != root_right:
        parent[root_right] = root_left


def _components(items, pairs):
    parent = {item: item for item in items}
    for left, right in pairs:
        _union(parent, left, right)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    groups = {}
    for item in items:
        groups.setdefault(find(item), []).append(item)
    return [members for members in groups.values() if len(members) > 1]


def detect_duplicate_artifacts(files, threshold=NEAR_DUPLICATE_THRESHOLD):
    """Detect exact code artifacts and >=threshold near-duplicate notebooks."""
    artifacts = []
    for name, path in files or []:
        try:
            artifact = _artifact_content(name, path)
        except Exception:
            artifact = None
        if artifact is not None:
            artifacts.append(artifact)

    order = {item["file"]: index for index, item in enumerate(artifacts)}
    by_name = {item["file"]: item for item in artifacts}
    exact_pairs = []
    for index, left in enumerate(artifacts):
        for right in artifacts[index + 1:]:
            if left["exact_hash"] == right["exact_hash"]:
                exact_pairs.append((left["file"], right["file"]))
    exact_groups = _components(list(by_name), exact_pairs)
    exact_members = {name for group in exact_groups for name in group}

    near_pairs = []
    near_scores = {}
    notebooks = [item for item in artifacts if item["suffix"] == ".ipynb" and item["file"] not in exact_members]
    for index, left in enumerate(notebooks):
        for right in notebooks[index + 1:]:
            similarity = SequenceMatcher(None, left["near_text"], right["near_text"], autojunk=False).ratio()
            if similarity >= threshold:
                pair = (left["file"], right["file"])
                near_pairs.append(pair)
                near_scores[frozenset(pair)] = similarity
    near_groups = _components([item["file"] for item in notebooks], near_pairs)

    groups = []
    file_status = {
        item["file"]: {
            "file": item["file"], "status": "Primary Analysis Artifact",
            "primary": item["file"], "duplicate_of": None, "similarity": 1.0,
        }
        for item in artifacts
    }
    for kind, members in (
        ("Duplicate Analysis Artifact", exact_groups),
        ("Near-duplicate Notebook", near_groups),
    ):
        for raw_members in members:
            ordered = sorted(raw_members, key=lambda value: order[value])
            primary = ordered[0]
            similarities = [
                near_scores.get(frozenset((left, right)), 1.0)
                for index, left in enumerate(ordered)
                for right in ordered[index + 1:]
            ]
            similarity = min(similarities) if similarities else 1.0
            groups.append({
                "type": kind,
                "primary": primary,
                "files": ordered,
                "similarity": similarity,
            })
            for name in ordered[1:]:
                file_status[name] = {
                    "file": name,
                    "status": kind,
                    "primary": primary,
                    "duplicate_of": primary,
                    "similarity": similarity,
                }

    return {
        "threshold": threshold,
        "groups": groups,
        "files": [file_status[item["file"]] for item in artifacts],
    }


def _normalized_snippet(issue):
    value = issue.get("source_snippet") or issue.get("evidence") or issue.get("message") or ""
    return re.sub(r"\s+", "", str(value)).casefold()


def _same_finding(left, right, group_type):
    if left.get("rule_id") != right.get("rule_id"):
        return False
    left_text, right_text = _normalized_snippet(left), _normalized_snippet(right)
    if group_type == "Duplicate Analysis Artifact":
        return left_text == right_text
    if not left_text or not right_text:
        return left.get("type") == right.get("type")
    return SequenceMatcher(None, left_text, right_text, autojunk=False).ratio() >= ISSUE_SNIPPET_THRESHOLD


def consolidate_duplicate_code_issues(code_result, artifact_analysis):
    """Merge repeated findings while retaining all affected files and locations."""
    groups = (artifact_analysis or {}).get("groups", [])
    membership = {
        name: (index, group)
        for index, group in enumerate(groups)
        for name in group.get("files", [])
    }
    consolidated = []
    candidates = {}
    for location in code_result or []:
        file_name = location.get("file", "Python file")
        membership_entry = membership.get(file_name)
        for issue in location.get("issues", []):
            copied = deepcopy(issue)
            copied.setdefault("affected_files", [file_name])
            copied.setdefault("affected_locations", [{
                "file": file_name,
                "cell": location.get("cell"),
                "line": issue.get("line"),
            }])
            match = None
            if membership_entry:
                group_index, group = membership_entry
                for candidate in candidates.get(group_index, []):
                    if _same_finding(candidate["issue"], copied, group.get("type")):
                        match = candidate
                        break
            if match:
                if file_name not in match["issue"]["affected_files"]:
                    match["issue"]["affected_files"].append(file_name)
                match["issue"]["affected_locations"].append(copied["affected_locations"][0])
                match["location"]["affected_files"] = list(match["issue"]["affected_files"])
                continue

            primary_file = (
                membership_entry[1].get("primary")
                if membership_entry and membership_entry[1].get("type") == "Duplicate Analysis Artifact"
                else file_name
            )
            target_location = {
                **location,
                "file": primary_file,
                "issues": [copied],
                "affected_files": list(copied["affected_files"]),
            }
            consolidated.append(target_location)
            if membership_entry:
                candidates.setdefault(membership_entry[0], []).append({
                    "issue": copied,
                    "location": target_location,
                })

    # Combine entries that point to the same primary file/cell without losing issue order.
    combined = []
    by_location = {}
    for location in consolidated:
        key = (location.get("file"), location.get("cell"))
        existing = by_location.get(key)
        if existing is None:
            by_location[key] = location
            combined.append(location)
        else:
            existing["issues"].extend(location.get("issues", []))
            existing["affected_files"] = list(dict.fromkeys(
                [*existing.get("affected_files", []), *location.get("affected_files", [])]
            ))
    return combined
