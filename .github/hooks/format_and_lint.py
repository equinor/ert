# ruff: file-ignore[implicit-namespace-package]
"""Format and lint Python files changed by a Copilot edit tool."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

PATCH_FILE_PATTERN = re.compile(r"\*\*\* (?:(?:Add|Update) File|Move to): ([^\n]+)")
MAX_FAILURE_OUTPUT_LENGTH = 2_000


def string_values(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested_value in value.values():
            yield from string_values(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            yield from string_values(nested_value)


def python_files(tool_args: object, repository_root: Path) -> list[Path]:
    if isinstance(tool_args, str):
        try:
            decoded_tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            decoded_tool_args = tool_args
        tool_args = decoded_tool_args

    paths: set[Path] = set()
    for value in string_values(tool_args):
        candidates = [value, *PATCH_FILE_PATTERN.findall(value)]
        for candidate in candidates:
            path = Path(candidate)
            if path.suffix != ".py":
                continue
            resolved_path = (
                path.resolve()
                if path.is_absolute()
                else (repository_root / path).resolve()
            )
            if (
                resolved_path.is_relative_to(repository_root)
                and resolved_path.is_file()
            ):
                paths.add(resolved_path)
    return sorted(paths)


def run_ruff(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def failure_context(results: list[subprocess.CompletedProcess[str]]) -> str:
    output = "\n".join(
        result.stderr or result.stdout for result in results if result.returncode != 0
    ).strip()
    return output[:MAX_FAILURE_OUTPUT_LENGTH] or "Ruff format or lint failed."


def main() -> None:
    payload = json.load(sys.stdin)
    repository_root = Path(payload["cwd"]).resolve()
    files = python_files(payload.get("toolArgs"), repository_root)
    if not files:
        print("{}")
        return

    file_arguments = [str(path.relative_to(repository_root)) for path in files]
    commands = [
        [
            "uv",
            "run",
            "--group",
            "style",
            "pre-commit",
            "run",
            hook,
            "--files",
            *file_arguments,
        ]
        for hook in ("ruff-check", "ruff-format")
    ]
    for command in commands:
        run_ruff(command)

    results = [run_ruff(command) for command in commands]
    if any(result.returncode != 0 for result in results):
        print(json.dumps({"additionalContext": failure_context(results)}))
    else:
        print("{}")


if __name__ == "__main__":
    main()
