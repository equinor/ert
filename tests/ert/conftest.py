import fileinput
import logging
import multiprocessing
import os
import resource
import shutil
import stat
import sys
import warnings
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
from textwrap import dedent

import polars as pl
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from PyQt6.QtCore import QDir
from PyQt6.QtWidgets import QApplication
from xlsxwriter import Workbook

import _ert.forward_model_runner.fm_dispatch
from _ert.threading import set_signal_handler
from ert.__main__ import ert_parser
from ert.cli.main import run_cli
from ert.config import ConfigWarning, ErtConfig
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.mode_definitions import (
    ENIF_MODE,
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
)
from ert.storage import open_storage

from .utils import SOURCE_DIR

st.register_type_strategy(Path, st.builds(Path, st.text().map(lambda x: "/tmp/" + x)))


@pytest.fixture(autouse=True)
def no_jobs_file_retry(monkeypatch):
    monkeypatch.setattr(_ert.forward_model_runner.fm_dispatch, "FILE_RETRY_TIME", 0)


@pytest.fixture(autouse=True)
def log_check():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    yield
    logger_after = logging.getLogger()
    level_after = logger_after.getEffectiveLevel()
    assert level_after == logging.WARNING, (
        f"Detected differences in log environment: Changed to {level_after}"
    )


@pytest.fixture(scope="session", autouse=True)
def _reraise_thread_exceptions_on_main_thread():
    """Allow `_ert.threading.ErtThread` to re-raise exceptions on main thread"""
    set_signal_handler()


@pytest.fixture
def _qt_add_search_paths(qapp):
    "Ensure that icons and such are found by the tests"
    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/gui/img")))


# Timeout settings are unreliable both on CI and
# when running pytest with xdist so we disable it
settings.register_profile(
    "no_timeouts",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
settings.load_profile("no_timeouts")

settings.register_profile(
    "fast",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
    max_examples=1,
)


@pytest.fixture(scope="session", name="source_root")
def fixture_source_root():
    return SOURCE_DIR


@pytest.fixture(scope="class")
def class_source_root(request, source_root):
    request.cls.SOURCE_ROOT = source_root
    request.cls.TESTDATA_ROOT = source_root / "test-data" / "ert"
    yield


@pytest.fixture(scope="session", autouse=True)
def maximize_ulimits():
    """
    Bumps the soft-limit for max number of files up to its max-value
    since we know that the tests may open lots of files simultaneously.
    Resets to original when session ends.
    """
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits[1], limits[1]))
    yield
    resource.setrlimit(resource.RLIMIT_NOFILE, limits)


@pytest.fixture(name="setup_case")
def fixture_setup_case(tmp_path_factory, source_root, monkeypatch):
    def copy_case(path, config_file):
        tmp_path = tmp_path_factory.mktemp(path.replace("/", "-"))
        shutil.copytree(
            os.path.join(source_root, "test-data/ert", path), tmp_path / "test_data"
        )
        monkeypatch.chdir(tmp_path / "test_data")
        return ErtConfig.from_file(config_file)

    yield copy_case


@pytest.fixture()
def poly_case(setup_case):
    return setup_case("poly_example", "poly.ert")


@pytest.fixture()
def snake_oil_case_storage(copy_snake_oil_case_storage):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConfigWarning)
        # Avoiding ConfigWarning on SUMMARY key with no known forward model
        return ErtConfig.from_file("snake_oil.ert")


@pytest.fixture()
def symlinked_snake_oil_case_storage(symlink_snake_oil_case_storage):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConfigWarning)
        # Avoiding ConfigWarning on SUMMARY key with no known forward model
        return ErtConfig.from_file("snake_oil.ert")


@pytest.fixture()
def symlinked_heat_equation_storage_es(symlink_heat_equation_storage_es):
    return ErtConfig.from_file("config.ert")


@pytest.fixture()
def symlinked_heat_equation_storage_esmda(symlink_heat_equation_storage_esmda):
    return ErtConfig.from_file("config.ert")


@pytest.fixture()
def symlinked_heat_equation_storage_enif(symlink_heat_equation_storage_enif):
    return ErtConfig.from_file("config.ert")


@pytest.fixture()
def snake_oil_case(setup_case):
    return setup_case("snake_oil", "snake_oil.ert")


@pytest.fixture()
def minimum_case(use_tmpdir):
    Path("minimum_config").write_text("NUM_REALIZATIONS 1", encoding="utf-8")
    return ErtConfig.from_file("minimum_config")


@pytest.fixture(name="copy_case")
def fixture_copy_case(tmp_path_factory, source_root, monkeypatch):
    def _copy_case(path):
        tmp_path = tmp_path_factory.mktemp(path.replace("/", "-"))
        shutil.copytree(
            os.path.join(source_root, "test-data/ert", path),
            tmp_path / "test_data",
            ignore=shutil.ignore_patterns("storage", "poly_out"),
        )
        monkeypatch.chdir(tmp_path / "test_data")

    yield _copy_case


def _create_design_matrix(filename, design_sheet_df, default_sheet_df=None):
    with Workbook(filename) as wb:
        design_sheet_df.write_excel(wb, worksheet="DesignSheet")
        if default_sheet_df is not None:
            default_sheet_df.write_excel(
                wb, worksheet="DefaultSheet", include_header=False
            )


@pytest.fixture()
def copy_poly_case(copy_case):
    copy_case("poly_example")
    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.write("QUEUE_OPTION LOCAL MAX_RUNNING 2\n")


@pytest.fixture()
def copy_poly_case_with_design_matrix(copy_case):
    def _create_poly_design_case(design_dict, default_list):
        copy_case("poly_example")
        num_realizations = len(design_dict["REAL"])
        _create_design_matrix(
            "poly_design.xlsx",
            pl.DataFrame(design_dict),
            pl.DataFrame(default_list, orient="row"),
        )
        Path("poly.ert").write_text(
            dedent(
                f"""\
                    QUEUE_OPTION LOCAL MAX_RUNNING 2
                    RUNPATH poly_out/realization-<IENS>/iter-<ITER>
                    NUM_REALIZATIONS {num_realizations}
                    MIN_REALIZATIONS 1
                    GEN_DATA POLY_RES RESULT_FILE:poly.out
                    DESIGN_MATRIX poly_design.xlsx DEFAULT_SHEET:DefaultSheet
                    INSTALL_JOB poly_eval POLY_EVAL
                    FORWARD_MODEL poly_eval
                    OBS_CONFIG observations
                    """
            ),
            encoding="utf-8",
        )

        Path("poly_eval.py").write_text(
            dedent(
                """\
                    #!/usr/bin/env python
                    import json

                    def _load_coeffs(filename):
                        with open(filename, encoding="utf-8") as f:
                            return json.load(f)

                    def _evaluate(coeffs, x):
                        return (coeffs["a"]["value"] * x**2 +
                                coeffs["b"]["value"] * x + coeffs["c"]["value"])

                    if __name__ == "__main__":
                        coeffs = _load_coeffs("parameters.json")
                        output = [_evaluate(coeffs, x) for x in range(10)]
                        with open("poly.out", "w", encoding="utf-8") as f:
                            f.write("\\n".join(map(str, output)))
                    """
            ),
            encoding="utf-8",
        )

        os.chmod(
            "poly_eval.py",
            os.stat("poly_eval.py").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    return _create_poly_design_case


@pytest.fixture()
def copy_snake_oil_field(copy_case):
    copy_case("snake_oil_field")
    with open("snake_oil_field.ert", "a", encoding="utf-8") as fh:
        fh.write("QUEUE_OPTION LOCAL MAX_RUNNING 2\n")


@pytest.fixture()
def copy_snake_oil_case(copy_case):
    copy_case("snake_oil")
    with open("snake_oil.ert", "a", encoding="utf-8") as fh:
        fh.write("QUEUE_OPTION LOCAL MAX_RUNNING 2\n")


@pytest.fixture()
def copy_heat_equation(copy_case):
    copy_case("heat_equation")
    with open("config.ert", "a", encoding="utf-8") as fh:
        fh.write("QUEUE_OPTION LOCAL MAX_RUNNING 2\n")


@pytest.fixture(
    name="copy_snake_oil_case_storage",
    params=[
        pytest.param(0, marks=pytest.mark.xdist_group(name="snake_oil_case_storage"))
    ],
)
def fixture_copy_snake_oil_case_storage(_shared_snake_oil_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_shared_snake_oil_case, "test_data")
    monkeypatch.chdir("test_data")


@pytest.fixture()
def symlink_snake_oil_case_storage(_shared_snake_oil_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.symlink(_shared_snake_oil_case, "test_data")
    monkeypatch.chdir("test_data")


@pytest.fixture
def symlink_heat_equation_storage_es(_shared_heat_equation_es, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.symlink(_shared_heat_equation_es, "heat_equation")
    monkeypatch.chdir("heat_equation")


@pytest.fixture
def symlink_heat_equation_storage_esmda(
    _shared_heat_equation_esmda, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    os.symlink(_shared_heat_equation_esmda, "heat_equation")
    monkeypatch.chdir("heat_equation")


@pytest.fixture
def symlink_heat_equation_storage_enif(
    _shared_heat_equation_enif, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    os.symlink(_shared_heat_equation_enif, "heat_equation_enif")
    monkeypatch.chdir("heat_equation_enif")


@pytest.fixture()
def copy_minimum_case(copy_case):
    copy_case("simple_config")


@pytest.fixture()
def use_tmpdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session", autouse=True)
def hide_window(request):
    if request.config.getoption("--show-gui"):
        yield
        return

    old_value = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    yield
    if old_value is None:
        del os.environ["QT_QPA_PLATFORM"]
    else:
        os.environ["QT_QPA_PLATFORM"] = old_value


@pytest.fixture
def _qt_excepthook(monkeypatch):
    """Hook into Python's unhandled exception handler and quit Qt if it's
    running. This will prevent a stall in the event that a Python exception
    occurs inside a Qt slot.

    """
    next_excepthook = sys.excepthook

    def excepthook(cls, exc, tb):
        if app := QApplication.instance():
            app.quit()
        next_excepthook(cls, exc, tb)

    monkeypatch.setattr(sys, "excepthook", excepthook)


def _run_snake_oil(source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data/ert", "snake_oil"), "test_data"
    )
    os.chdir("test_data")
    with fileinput.input("snake_oil.ert", inplace=True) as fin:
        for line in fin:
            if "NUM_REALIZATIONS 25" in line:
                print("NUM_REALIZATIONS 5", end="")
            else:
                print(line, end="")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "--disable-monitoring",
            "--current-ensemble",
            "default_0",
            "snake_oil.ert",
        ],
    )

    run_cli(parsed)


def _run_heat_equation(source_root, run_mode):
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "heat_equation"), "test_data"
    )
    os.chdir("test_data")
    with open("config.ert", "a", encoding="utf-8") as fh:
        fh.write("QUEUE_OPTION LOCAL MAX_RUNNING 2\n")
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            run_mode,
            "--disable-monitoring",
            "config.ert",
        ],
    )

    run_cli(parsed)


@pytest.fixture
def _shared_snake_oil_case(request, monkeypatch, source_root):
    """This fixture will run the snake_oil case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "snake_oil_data" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.path.exists(snake_path / "test_data"):
        _run_snake_oil(source_root)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def _shared_heat_equation_es(request, monkeypatch, source_root):
    """This fixture will run the heat_equation case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "heat_equation_data_es" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.listdir(snake_path):
        _run_heat_equation(source_root, ENSEMBLE_SMOOTHER_MODE)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def _shared_heat_equation_esmda(request, monkeypatch, source_root):
    """This fixture will run the heat_equation case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "heat_equation_data_esmda" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.listdir(snake_path):
        _run_heat_equation(source_root, ES_MDA_MODE)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def _shared_heat_equation_enif(request, monkeypatch, source_root):
    """This fixture will run the heat_equation case to populate storage,
    this is quite slow, but the results will be cached. If something comes
    out of sync, clear the cache and start again.
    """
    snake_path = request.config.cache.mkdir(
        "heat_equation_data_enif" + os.environ.get("PYTEST_XDIST_WORKER", "")
    )
    monkeypatch.chdir(snake_path)
    if not os.listdir(snake_path):
        _run_heat_equation(source_root, ENIF_MODE)
    else:
        monkeypatch.chdir("test_data")

    yield os.getcwd()


@pytest.fixture
def storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        yield storage


@pytest.fixture
def new_ensemble(storage):
    experiment_id = storage.create_experiment()
    return storage.create_ensemble(
        experiment_id, name="new_ensemble", ensemble_size=100
    )


@pytest.fixture
def snake_oil_storage(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ens_path, mode="w") as storage:
        yield storage


@pytest.fixture
def snake_oil_default_storage(snake_oil_case_storage):
    with open_storage(snake_oil_case_storage.ens_path) as storage:
        experiment = storage.get_experiment_by_name("ensemble-experiment")
        yield experiment.get_ensemble_by_name("default_0")


@pytest.fixture(scope="session")
def block_storage_path(source_root):
    path = source_root / "test-data/ert/block_storage/snake_oil"
    if not path.is_dir():
        pytest.skip(
            "'test-data/ert/block_storage' has not been checked out.\n"
            "Make sure you have git-lfs installed and run: "
            "git submodule update --init --recursive"
        )
    return path.parent


@pytest.fixture(autouse=True)
def no_cert_in_test(monkeypatch):
    # Do not generate certificates during test, parts of it can be time
    # consuming (e.g. 30 seconds)
    # Specifically generating the RSA key <_openssl.RSA_generate_key_ex>
    class MockESConfig(EvaluatorServerConfig):
        def __init__(self, *args, **kwargs) -> None:
            if "use_token" not in kwargs:
                kwargs["use_token"] = False
            if sys.platform != "linux":
                kwargs["use_ipc_protocol"] = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("ert.cli.main.EvaluatorServerConfig", MockESConfig)


@pytest.fixture(autouse=True)
def no_hostname_on_mac(monkeypatch):
    def mock_get_machine_name() -> str:
        return "localhost"

    if sys.platform != "linux":
        monkeypatch.setattr(
            "ert.ensemble_evaluator.evaluator.get_machine_name", mock_get_machine_name
        )


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_method():
    if (
        sys.platform == "linux"
        and multiprocessing.get_start_method(allow_none=True) != "forkserver"
    ):
        multiprocessing.set_start_method("forkserver")
