"""Line-preserving preprocessing for IPython-only notebook syntax."""

from __future__ import annotations

import re
import io
import tokenize


NON_PYTHON_CELL_MAGICS = {
    "bash", "sh", "script", "html", "javascript", "js", "latex", "sql", "ruby", "perl",
}


def _python_statement_is_open(lines):
    """Return whether ``lines`` end inside an unfinished Python statement.

    IPython permits a line magic at the start of a *new* statement, but ``%``
    is also Python's modulo/string-formatting operator.  In particular, code
    such as ``("%s"\n % value)`` is valid Python and must not be stripped.
    Tokenization is a better fit than indentation heuristics because it also
    covers explicit continuations and multi-line strings while remaining
    independent of whether earlier complete statements can be executed.
    """
    prefix = "\n".join(lines)
    if not prefix.strip():
        return False
    try:
        list(tokenize.generate_tokens(io.StringIO(prefix + "\n").readline))
    except (tokenize.TokenError, IndentationError) as exc:
        message = str(exc).casefold()
        return any(marker in message for marker in (
            "multi-line statement", "multi-line string", "eof in multi-line",
        ))
    # A backslash continuation can be tokenized successfully after a newline
    # is appended, so retain an explicit check for the physical last line.
    return bool(lines and re.search(r"(?<!\\)\\\s*$", lines[-1]))


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
        statement_open = _python_statement_is_open(processed[:index - 1])
        if stripped.startswith("%") and not statement_open:
            constructs.append({
                "kind": "IPython Magic",
                "subtype": "line_magic",
                "line": index,
                "source": line,
            })
            processed[index - 1] = ""
        elif stripped.startswith("!") and not statement_open:
            constructs.append({
                "kind": "Shell Command",
                "subtype": "shell_command",
                "line": index,
                "source": line,
            })
            processed[index - 1] = ""
        elif (not statement_open and stripped.startswith("?")) or (
            not statement_open
            and
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
