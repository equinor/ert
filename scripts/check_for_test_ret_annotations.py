#!/usr/bin/env python3
"""Fail if any pytest-collected test function declares '-> None'.

This is used as a linting rule in .pre-commit.yml
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path


def _is_none_literal(node: ast.expr | None) -> bool:
    # `-> None` is always ast.Constant(value=None), regardless of
    # `from __future__ import annotations`.
    return isinstance(node, ast.Constant) and node.value is None


def _iter_test_functions(
    tree: ast.Module,
) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Module-level functions and methods of Test* classes (pytest's defaults)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield sub


def _check(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        print(f"{path}: could not parse: {exc}", file=sys.stderr)
        return []
    return [
        (node.lineno, node.name)
        for node in _iter_test_functions(tree)
        if node.name.startswith("test_") and _is_none_literal(node.returns)
    ]


def main(paths: list[str]) -> int:
    if not paths:
        paths = [str(p) for p in Path("tests").rglob("*.py")]
    found = False
    for p in paths:
        path = Path(p)
        for lineno, name in _check(path):
            print(f"{path}:{lineno}: test function '{name}' has a redundant '-> None'")
            found = True
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
