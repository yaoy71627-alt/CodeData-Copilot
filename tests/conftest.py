"""Offline regression: never load a local secret or send a paid model request."""
import os
from unittest.mock import patch

# Set before test-module collection imports agent/app and their load_dotenv().
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import socket
import dotenv
import pytest
import hashlib
from pathlib import Path
import shutil

# Older python-dotenv versions do not recognize PYTHON_DOTENV_DISABLED.
# Patch before collecting modules so no supported version reads .env files.
_dotenv_guard = patch.object(dotenv, "load_dotenv", return_value=False)
_dotenv_guard.start()


def pytest_unconfigure(config):
    _dotenv_guard.stop()


def pytest_addoption(parser):
    parser.addoption("--online", action="store_true", default=False,
                     help="Explicitly allow e2e tests to call the configured paid API")


@pytest.fixture(autouse=True)
def offline_model_transport(monkeypatch, request):
    """Unexpected SDK traffic fails even when the application catches the error."""
    if request.node.get_closest_marker("e2e"):
        if not request.config.getoption("--online"):
            pytest.skip("Online E2E requires explicit --online and a local environment API key")
        if not os.environ.get("DASHSCOPE_API_KEY"):
            pytest.fail("Online E2E requires DASHSCOPE_API_KEY set in the process environment")
        yield
        return
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Regression tests require a mocked HTTP transport")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "regression-placeholder-not-a-key")
    for key in list(os.environ):
        if key.startswith("DASHSCOPE_") and key != "DASHSCOPE_API_KEY":
            monkeypatch.delenv(key)
    # Transport independent: SDK versions can use different HTTP libraries.
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    yield
    assert not attempts, "An unmocked HTTP request was blocked; fix the test's mock"


@pytest.fixture(scope="module", params=["titanic", "cardekho"])
def real_case_files(request, tmp_path_factory):
    """Explicit local sources, never silently substitute constructed data."""
    name = request.param
    prefix = "CODEDATA_" + name.upper()
    notebook, csv = os.environ.get(prefix + "_NOTEBOOK"), os.environ.get(prefix + "_CSV")
    if not notebook and not csv:
        pytest.skip(f"Original {name} files not configured: set {prefix}_NOTEBOOK and {prefix}_CSV")
    if not notebook or not csv:
        pytest.fail(f"Both {prefix}_NOTEBOOK and {prefix}_CSV are required")
    paths = [Path(notebook), Path(csv)]
    if not all(p.is_file() for p in paths):
        pytest.fail(f"Configured original {name} files do not exist")
    root = tmp_path_factory.mktemp("real_" + name)
    # Filename aliases only; original notebook/data bytes remain unchanged.
    aliases = [paths[0].name, "train_问题案例.csv" if name == "titanic" else "cardekho_dataset.csv"]
    files, hashes = [], {}
    for original, alias in zip(paths, aliases):
        local = root / alias
        shutil.copyfile(original, local)
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
        files.append((alias, local))
        hashes[alias] = digest
    return name, files, hashes
