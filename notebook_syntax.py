"""Line-preserving preprocessing for IPython-only notebook syntax."""

from __future__ import annotations

import re


NON_PYTHON_CELL_MAGICS = {
    "bash", "sh", "script", "html", "javascript", "js", "latex", "sql", "ruby", "perl",
}


def preprocess_notebook_source(source):
    """Replace notebook-only syntax with blank lines and retain original line numbers."""
    original_lines = str(source).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    processed = list(original_lines)
    constructs = []
    cell_magic = None

    for index, line in enumerate(original_lines, 1):
        stripped = line.lstrip()
        if index == 1 and stripped.startswith("%%"):
            match = re.match(r"%%\s*([A-Za-z_]\w*)", stripped)
            cell_magic = match.group(1).casefold() if match else "unknown"
            constructs.append({
                "kind": "IPython Magic",
                "subtype": "cell_magic",
                "name": cell_magic,
                "line": index,
                "source": line,
            })
            processed[index - 1] = ""
            continue
        if cell_magic in NON_PYTHON_CELL_MAGICS:
            processed[index - 1] = ""
            continue
        if stripped.startswith("%"):
            constructs.append({
                "kind": "IPython Magic",
                "subtype": "line_magic",
                "line": index,
                "source": line,
            })
            processed[index - 1] = ""
        elif stripped.startswith("!"):
            constructs.append({
                "kind": "Shell Command",
                "subtype": "shell_command",
                "line": index,
                "source": line,
            })
            processed[index - 1] = ""
        elif stripped.startswith("?") or (
            stripped.endswith("?")
            and re.match(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\?\s*$", stripped)
        ):
            constructs.append({
                "kind": "Notebook-specific Syntax",
                "subtype": "help",
                "line": index,
                "source": line,
            })
            processed[index - 1] = ""

    return {
        "code": "\n".join(processed),
        "constructs": constructs,
        "line_map": {index: index for index in range(1, len(original_lines) + 1)},
    }
