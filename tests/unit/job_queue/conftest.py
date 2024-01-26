import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ert
from ert.load_status import LoadStatus


@pytest.fixture
def mock_fm_ok(monkeypatch):
    fm_ok = MagicMock(return_value=(LoadStatus.LOAD_SUCCESSFUL, ""))
    monkeypatch.setattr(ert.job_queue.job_queue_node, "forward_model_ok", fm_ok)
    yield fm_ok


@pytest.fixture
def simple_script(tmp_path):
    SIMPLE_SCRIPT = """#!/bin/sh
echo "finished successfully" > STATUS
"""
    fout = Path(tmp_path / "job_script")
    fout.write_text(SIMPLE_SCRIPT, encoding="utf-8")
    fout.chmod(stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)
    yield str(fout)


@pytest.fixture
def failing_script(tmp_path):
    """
    This script is susceptible to race conditions. Python works
    better than sh."""
    FAILING_SCRIPT = """#!/usr/bin/env python
import sys
with open("one_byte_pr_invocation", "a") as f:
    f.write(".")
sys.exit(1)
    """
    fout = Path(tmp_path / "failing_script")
    fout.write_text(FAILING_SCRIPT, encoding="utf-8")
    fout.chmod(stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)
    yield str(fout)
