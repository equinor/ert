import pytest
from hypothesis import settings

from ert.config import ert_config

settings.register_profile("blobs", print_blob=True)
settings.load_profile("blobs")


@pytest.fixture()
def set_site_config(monkeypatch, tmp_path):
    test_site_config = tmp_path / "test_site_config.ert"
    test_site_config.write_text("JOB_SCRIPT job_dispatch.py\nQUEUE_SYSTEM LOCAL\n")
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))


@pytest.fixture(autouse=True)
def do_not_append_load_results_job(monkeypatch):
    monkeypatch.setattr(ert_config, "APPEND_LOAD_RESULTS_JOB", False)
