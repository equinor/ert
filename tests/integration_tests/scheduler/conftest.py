from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def mock_bin(monkeypatch, tmp_path):
    bin_path = Path(__file__).parent / "bin"

    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTEST_TMP_PATH", str(tmp_path))
    monkeypatch.setenv("PYTHON", sys.executable)


@pytest.fixture
def job_name(request) -> str:
    return request.node.name.split("[")[0]
