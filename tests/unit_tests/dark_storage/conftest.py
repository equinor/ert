import contextlib
import gc
import os
import shutil
from argparse import ArgumentParser

import pytest
from py import path
from starlette.testclient import TestClient

from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.dark_storage import enkf
from ert.storage import Mode, Storage


@pytest.fixture(scope="session")
def poly_example_tmp_dir_shared(
    tmp_path_factory,
    source_root,
):
    tmpdir = tmp_path_factory.mktemp("my_poly_tmp")
    poly_dir = path.local(os.path.join(str(tmpdir), "poly_example"))
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
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
            ],
        )
        run_cli(parsed)
    return poly_dir


@pytest.fixture()
def poly_example_tmp_dir(poly_example_tmp_dir_shared):
    with poly_example_tmp_dir_shared.as_cwd():
        yield


@pytest.fixture()
def fresh_storage(tmp_path_factory):
    tmp = path.local(tmp_path_factory.mktemp("atmp"))
    with tmp.as_cwd():
        storage = Storage(tmp / "storage", mode=Mode.WRITE)
        yield storage


@pytest.fixture
def dark_storage_client(monkeypatch):
    with dark_storage_app_(monkeypatch) as dark_app:
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", "storage")
        with TestClient(dark_app) as client:
            yield client


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")


def reset_enkf():
    if enkf._storage is not None:
        enkf._storage.close()
    enkf._storage = None
    gc.collect()


@contextlib.contextmanager
def dark_storage_app_(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
    monkeypatch.setenv("ERT_STORAGE_ENS_PATH", "storage")
    from ert.dark_storage.app import app

    yield app
    reset_enkf()


@pytest.fixture
def dark_storage_app(monkeypatch):
    with dark_storage_app_(monkeypatch) as app:
        yield app
