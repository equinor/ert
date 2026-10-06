from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_hook_module() -> ModuleType:
    hook_path = Path(__file__).parents[3] / ".github/hooks/format_and_lint.py"
    spec = importlib.util.spec_from_file_location("format_and_lint", hook_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_that_python_files_extracts_existing_repository_python_files(tmp_path):
    hook = load_hook_module()
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    direct_file = repository_root / "direct.py"
    direct_file.touch()
    patched_file = repository_root / "nested/patched.py"
    patched_file.parent.mkdir()
    patched_file.touch()
    outside_file = tmp_path / "outside.py"
    outside_file.touch()

    tool_args = json.dumps(
        {
            "path": "direct.py",
            "patch": "*** Add File: nested/patched.py\n+content\n",
            "outside": str(outside_file),
            "missing": "missing.py",
            "non_python": "README.md",
        }
    )

    assert hook.python_files(tool_args, repository_root) == [
        direct_file,
        patched_file,
    ]


def test_that_main_runs_only_ruff_hooks_twice_for_changed_python_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    hook = load_hook_module()
    python_file = tmp_path / "changed.py"
    python_file.touch()
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(json.dumps({"cwd": str(tmp_path), "toolArgs": "changed.py"})),
    )
    commands: list[list[str]] = []

    def run_hook(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(hook, "run_ruff", run_hook)

    hook.main()

    assert commands == [
        [
            "uv",
            "run",
            "--group",
            "style",
            "pre-commit",
            "run",
            hook_name,
            "--files",
            "changed.py",
        ]
        for hook_name in ("ruff-check", "ruff-format") * 2
    ]
    assert json.loads(capsys.readouterr().out) == {}


def test_that_main_ignores_file_modifications_when_verification_passes(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    hook = load_hook_module()
    python_file = tmp_path / "changed.py"
    python_file.touch()
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(json.dumps({"cwd": str(tmp_path), "toolArgs": "changed.py"})),
    )
    results = iter(
        [
            subprocess.CompletedProcess([], 1, "fixed lint", ""),
            subprocess.CompletedProcess([], 1, "formatted", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )
    monkeypatch.setattr(hook, "run_ruff", lambda _command: next(results))

    hook.main()

    assert json.loads(capsys.readouterr().out) == {}
