from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_install_script():
    spec = importlib.util.spec_from_file_location(
        "context_rag_install_script", REPO_ROOT / "scripts" / "install.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_runs_editable_pip_install(monkeypatch):
    script = load_install_script()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="usage: context-rag", stderr="")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    assert script.main([]) == 0
    assert calls[0] == [sys.executable, "-m", "pip", "install", "-e", "."]
    assert calls[1] == [sys.executable, "-m", "context_rag.cli", "--help"]


def test_dev_flag_uses_dev_extra_when_available(monkeypatch):
    script = load_install_script()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="usage: context-rag", stderr="")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    assert script.main(["--dev"]) == 0
    assert calls[0] == [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]


def test_dry_run_does_not_call_subprocess(monkeypatch):
    script = load_install_script()

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not call subprocess.run")

    monkeypatch.setattr(script.subprocess, "run", fail_run)

    assert script.main(["--dry-run"]) == 0
