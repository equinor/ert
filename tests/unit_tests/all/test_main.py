from unittest.mock import Mock

import pytest
from packaging.version import Version

import ert.shared
from ert.__main__ import ert_parser
from ert.cli import (
    ENSEMBLE_EXPERIMENT_MODE,
    ENSEMBLE_SMOOTHER_MODE,
    ES_MDA_MODE,
    ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
    TEST_RUN_MODE,
    WORKFLOW_MODE,
)


@pytest.fixture(autouse=True)
def mocked_valid_file(monkeypatch):
    monkeypatch.setattr(
        ert.__main__, "valid_file", Mock(return_value="path/to/config.ert")
    )


def test_argparse_exec_gui():
    parsed = ert_parser(None, ["gui", "path/to/config.ert"])
    assert parsed.func.__name__ == "run_gui_wrapper"


@pytest.mark.parametrize("input_path", ["a/path/config.ert", "another/path/config.ert"])
def test_parsed_config(monkeypatch, input_path):
    monkeypatch.setattr(
        ert.__main__, "valid_file", Mock(side_effect=lambda x: input_path)
    )
    parsed = ert_parser(None, [TEST_RUN_MODE, input_path])
    assert parsed.config == input_path


def test_argparse_exec_test_run_valid_case():
    parsed = ert_parser(None, [TEST_RUN_MODE, "path/to/config.ert"])
    assert parsed.mode == TEST_RUN_MODE
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_ensemble_experiment_valid_case():
    parsed = ert_parser(
        None,
        [ENSEMBLE_EXPERIMENT_MODE, "--realizations", "1-4,7,8", "path/to/config.ert"],
    )
    assert parsed.mode == ENSEMBLE_EXPERIMENT_MODE
    assert parsed.realizations == "1-4,7,8"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_ensemble_experiment_current_case():
    parsed = ert_parser(
        None,
        [ENSEMBLE_EXPERIMENT_MODE, "--current-case", "test_case", "path/to/config.ert"],
    )
    assert parsed.mode == ENSEMBLE_EXPERIMENT_MODE
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_ensemble_experiment_faulty_realizations():
    with pytest.raises(SystemExit):
        ert_parser(
            None,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--realizations",
                "1~4,7,",
                "path/to/config.ert",
            ],
        )


def test_argparse_exec_ensemble_smoother_valid_case():
    parsed = ert_parser(
        None,
        [ENSEMBLE_SMOOTHER_MODE, "--target-case", "some_case", "path/to/config.ert"],
    )
    assert parsed.mode == ENSEMBLE_SMOOTHER_MODE
    assert parsed.target_case == "some_case"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_ensemble_smoother_no_target_case():
    with pytest.raises(SystemExit):
        ert_parser(None, [ENSEMBLE_SMOOTHER_MODE, "path/to/config.ert"])


def test_argparse_exec_iterative_ensemble_smoother_valid_case():
    parsed = ert_parser(
        None,
        [
            ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
            "--target-case",
            "some_case_%d",
            "path/to/config.ert",
        ],
    )
    assert parsed.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE
    assert parsed.target_case == "some_case_%d"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_iterative_ensemble_smoother_no_target_case():
    with pytest.raises(SystemExit):
        ert_parser(None, [ITERATIVE_ENSEMBLE_SMOOTHER_MODE, "path/to/config.ert"])


def test_argparse_exec_es_mda_valid_case():
    parsed = ert_parser(
        None,
        [
            ES_MDA_MODE,
            "--target-case",
            "some_case%d",
            "--realizations",
            "1-10",
            "--weights",
            "1, 2, 4",
            "path/to/config.ert",
        ],
    )
    assert parsed.mode == ES_MDA_MODE
    assert parsed.target_case == "some_case%d"
    assert parsed.realizations == "1-10"
    assert parsed.weights == "1, 2, 4"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_es_mda_default_weights():
    parsed = ert_parser(None, [ES_MDA_MODE, "path/to/config.ert"])
    assert parsed.mode == ES_MDA_MODE
    assert parsed.weights == "4, 2, 1"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_ensemble_es_mda_restart_case():
    parsed = ert_parser(
        None, [ES_MDA_MODE, "--restart-case", "test_case", "path/to/config.ert"]
    )
    assert parsed.mode == ES_MDA_MODE
    assert parsed.restart_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_workflow():
    parsed = ert_parser(None, [WORKFLOW_MODE, "workflow_name", "path/to/config.ert"])
    assert parsed.mode == WORKFLOW_MODE
    assert parsed.name == "workflow_name"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_ensemble_smoother_current_case():
    parsed = ert_parser(
        None,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "test_case",
            "--target-case",
            "test_case_smoother",
            "path/to/config.ert",
        ],
    )
    assert parsed.mode == ENSEMBLE_SMOOTHER_MODE
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_argparse_exec_iterative_ensemble_smoother_current_case():
    parsed = ert_parser(
        None,
        [
            ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "test_case",
            "--target-case",
            "test_case_smoother_%d",
            "path/to/config.ert",
        ],
    )
    assert parsed.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


@pytest.mark.parametrize(
    "flag_val,expected_value", [("--verbose", True), (None, False)]
)
def test_verbose_flag(flag_val, expected_value):
    args = [TEST_RUN_MODE, " path/to/config.ert"]
    if flag_val:
        args.append(flag_val)
    parsed = ert_parser(None, args)
    assert parsed.verbose == expected_value


def test_version_valid_Version(capsys):
    try:
        ert_parser(None, ["--version"])
    except SystemExit as e:
        assert e.code == 0

    ert_version, _ = capsys.readouterr()
    ert_version = ert_version.rstrip("\n")

    assert Version(ert_version)


def test_version_mocked(capsys, monkeypatch):
    monkeypatch.setattr(ert.shared, "__version__", "1.0.3")

    try:
        ert_parser(None, ["--version"])
    except SystemExit as e:
        assert e.code == 0

    ert_version, _ = capsys.readouterr()
    ert_version = ert_version.rstrip("\n")

    assert ert_version == "1.0.3"


@pytest.mark.parametrize(
    "port_input,expected_range",
    [("10-20", range(10, 21)), ("0-65535", range(0, 65536)), ("1-1", range(1, 2))],
)
def test_argparse_valid_port_range(port_input, expected_range):
    parsed = ert_parser(
        None,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "--port-range",
            port_input,
            "path/to/config.ert",
        ],
    )
    assert parsed.port_range == expected_range


@pytest.mark.parametrize(
    "port_input",
    [("20-10"), ("65535"), ("1--31"), ("0-65536")],
)
def test_argparse_invalid_port_range(
    port_input,
):
    with pytest.raises(SystemExit):
        ert_parser(
            None,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--port-range",
                port_input,
                "path/to/config.ert",
            ],
        )


def test_argparse_no_port_range():
    parsed = ert_parser(
        None,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "path/to/config.ert",
        ],
    )
    assert parsed.port_range is None
