import contextlib
from argparse import ArgumentParser

import pytest
import os
import shutil
import py
import gc

from starlette.testclient import TestClient

from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.main import run_cli
from ert_shared.dark_storage import enkf
from ert_shared.main import ert_parser


@pytest.fixture(scope="session")
def poly_example_tmp_dir_shared(
    tmp_path_factory,
    source_root,
):
    tmpdir = tmp_path_factory.mktemp("my_poly_tmp")
    poly_dir = py.path.local(os.path.join(str(tmpdir), "poly_example"))
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        poly_dir,
    )
    with poly_dir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "alpha",
                "--target-case",
                "beta",
                "--realizations",
                "1,2,4",
                "poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        run_cli(parsed)
    return poly_dir


@pytest.fixture()
def poly_example_tmp_dir(poly_example_tmp_dir_shared):
    with poly_example_tmp_dir_shared.as_cwd():
        yield


@pytest.fixture
def dark_storage_client(monkeypatch):
    with dark_storage_app_(monkeypatch) as dark_app:
        monkeypatch.setenv("ERT_STORAGE_RES_CONFIG", "poly.ert")
        yield TestClient(dark_app)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")


@pytest.fixture
def ert_storage_app(env):
    from ert_storage.app import app

    return app


def reset_enkf():
    enkf.ids = {}
    enkf._config = None
    enkf._ert = None
    enkf._libres_facade = None
    gc.collect()


def new_storage_client(monkeypatch, ert_storage_client):
    import requests
    from ert_shared.storage import extraction

    class MockStorage:
        @staticmethod
        def session():
            return ert_storage_client

    monkeypatch.setattr(extraction, "Storage", MockStorage)

    return ert_storage_client


@pytest.fixture
def run_poly_example_new_storage(monkeypatch, tmpdir, source_root):
    poly_dir = py.path.local(os.path.join(str(tmpdir), "poly_example"))
    shutil.copytree(
        os.path.join(source_root, "test-data", "local", "poly_example"),
        poly_dir,
    )
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
    monkeypatch.setenv("ERT_STORAGE_RES_CONFIG", "poly.ert")
    monkeypatch.setenv("ERT_STORAGE_DATABASE_URL", "sqlite:///:memory:")

    from ert_storage.testing.testclient import testclient_factory

    with poly_dir.as_cwd(), testclient_factory() as ert_storage_client, dark_storage_app_(
        monkeypatch
    ) as dark_app:
        new_storage_client_ = new_storage_client(monkeypatch, ert_storage_client)

        from ert_shared.feature_toggling import FeatureToggling

        # Enable new storage
        feature = FeatureToggling._conf["new-storage"]
        monkeypatch.setattr(feature, "is_enabled", True)
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,3,5",
                "poly.ert",
                "--port-range",
                "1024-65535",
            ],
        )
        run_cli(parsed)

        dark_storage_client_ = TestClient(dark_app)

        yield new_storage_client_, dark_storage_client_


@contextlib.contextmanager
def dark_storage_app_(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
    monkeypatch.setenv("ERT_STORAGE_RES_CONFIG", "poly.ert")
    monkeypatch.setenv("ERT_STORAGE_DATABASE_URL", "sqlite://")
    from ert_shared.dark_storage.app import app

    yield app
    reset_enkf()


@pytest.fixture
def dark_storage_app(monkeypatch):
    with dark_storage_app_(monkeypatch) as app:
        yield app
