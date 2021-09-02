import pytest
import os
import shutil
import py

from starlette.testclient import TestClient
from ert_shared.dark_storage import enkf


@pytest.fixture
def poly_example_tmp_dir(tmpdir, source_root, monkeypatch):
    monkeypatch.setenv("ERT_DARK_STORAGE_CONFIG", "poly.ert")
    poly_dir = py.path.local(os.path.join(str(tmpdir), "poly_example"))
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        poly_dir,
    )

    with poly_dir.as_cwd():
        yield


@pytest.fixture
def dark_storage_client(dark_storage_app):
    return TestClient(dark_storage_app)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_DATABASE_URL", "sqlite://")
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")


@pytest.fixture
def ert_storage_app(env):
    from ert_storage.app import app

    return app


@pytest.fixture
def dark_storage_app(env):
    from ert_shared.dark_storage.app import app

    yield app
    reset_enkf()


def reset_enkf():
    enkf.ids = {}
    enkf._config = None
    enkf._ert = None
    enkf._libres_facade = None
