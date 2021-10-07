from argparse import ArgumentParser

import pytest
import os
import shutil
import py

from starlette.testclient import TestClient

from ert_shared.cli import ENSEMBLE_SMOOTHER_MODE
from ert_shared.cli.main import run_cli
from ert_shared.dark_storage import enkf
from ert_shared.main import ert_parser


@pytest.fixture(scope="session")
def poly_example_tmp_dir(
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
                "--target-case",
                "poly_runpath_file",
                "--realizations",
                "1,2,4",
                "poly.ert",
            ],
        )

        run_cli(parsed)
        yield


@pytest.fixture
def dark_storage_client(dark_storage_app, monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_RES_CONFIG", "poly.ert")
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
