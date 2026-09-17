import contextlib
import gc
import shutil
from argparse import ArgumentParser

import pytest
from py import path
from starlette.testclient import TestClient

from ert.__main__ import ert_parser
from ert.cli.main import run_cli
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.server import common
from ert.server.app import app


@pytest.fixture(scope="session")
def poly_example_tmp_dir_shared(
    tmp_path_factory,
    source_root,
):
    tmpdir = tmp_path_factory.mktemp("my_poly_tmp")
    poly_dir = path.local(tmpdir / "poly_example")
    shutil.copytree(
        source_root / "test-data" / "ert" / "poly_example",
        poly_dir,
        ignore=shutil.ignore_patterns("*ipynb", "poly_out", "storage", "logs"),
    )
    with poly_dir.as_cwd():
        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--disable-monitoring",
                "--realizations",
                "1,2,4",
                "poly.ert",
            ],
        )
        run_cli(parsed)
    return poly_dir


@pytest.fixture
def poly_example_tmp_dir(poly_example_tmp_dir_shared):
    with poly_example_tmp_dir_shared.as_cwd():
        yield


@pytest.fixture
def server_client(monkeypatch):
    with server_app_(monkeypatch) as server_app_instance:
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", "storage")
        with TestClient(server_app_instance) as client:
            yield client


@pytest.fixture
def server_client_snake_oil(monkeypatch):
    with server_app_(monkeypatch) as server_app_instance:
        monkeypatch.setenv("ERT_STORAGE_ENS_PATH", "storage/snake_oil/ensemble")
        with TestClient(server_app_instance) as client:
            yield client


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")


def reset_server():
    if common._storage is not None:
        common._storage.close()
    common._storage = None
    gc.collect()


@contextlib.contextmanager
def server_app_(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")
    monkeypatch.setenv("ERT_STORAGE_ENS_PATH", "storage")

    try:
        yield app
    finally:
        reset_server()


@pytest.fixture
def server_app(monkeypatch):
    with server_app_(monkeypatch) as app:
        yield app
