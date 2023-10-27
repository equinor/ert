import pytest

from ert.config import ert_config


@pytest.fixture(autouse=True)
def do_not_append_load_results_job(monkeypatch):
    monkeypatch.setattr(ert_config, "APPEND_LOAD_RESULTS_JOB", False)
