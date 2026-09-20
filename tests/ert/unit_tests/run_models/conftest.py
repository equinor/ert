import pytest


@pytest.fixture
def create_dummy_runpath(tmp_path, monkeypatch):
    runpath = tmp_path / "out"
    (runpath / "realization-0" / "iter-0").mkdir(parents=True)
    (runpath / "realization-1" / "iter-0").mkdir(parents=True)
    (runpath / "realization-1" / "iter-1").mkdir(parents=True)
    return monkeypatch.chdir(tmp_path)
