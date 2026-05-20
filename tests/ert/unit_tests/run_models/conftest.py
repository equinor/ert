import pytest


@pytest.fixture
def create_dummy_run_path(tmp_path, monkeypatch):
    run_path = tmp_path / "out"
    (run_path / "realization-0" / "iter-0").mkdir(parents=True)
    (run_path / "realization-1" / "iter-0").mkdir(parents=True)
    (run_path / "realization-1" / "iter-1").mkdir(parents=True)
    return monkeypatch.chdir(tmp_path)
