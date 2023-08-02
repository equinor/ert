import pytest
from hypothesis import settings

settings.register_profile("blobs", print_blob=True)
settings.load_profile("blobs")


@pytest.fixture()
def set_site_config(monkeypatch, tmp_path):
    test_site_config = tmp_path / "test_site_config.ert"
    test_site_config.write_text("JOB_SCRIPT job_dispatch.py\nQUEUE_SYSTEM LOCAL\n")
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))
