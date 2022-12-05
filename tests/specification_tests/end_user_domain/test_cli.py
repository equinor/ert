"""
Tests CLI behavior expected by an ert user.
"""
from pathlib import Path
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
from ert.cli.main import ErtCliError, run_cli


@pytest.fixture(autouse=True)
def mocked_valid_file(monkeypatch):
    # GIVEN a valid user config file named config.ert
    monkeypatch.setattr(ert.__main__, "valid_file", Mock(return_value="config.ert"))


def test_that_gui_can_be_started_via_cli():
    # WHEN I select to run the gui via cli
    parsed = ert_parser(None, ["gui", "config.ert"])

    # THEN I expect ert to run the gui
    assert parsed.func.__name__ == "run_gui_wrapper"


def test_that_test_run_can_be_selected_via_cli():
    # WHEN I select to run a test run via the cli
    parsed = ert_parser(None, [TEST_RUN_MODE, "config.ert"])

    # THEN I expect ert to run a test run in the cli
    assert parsed.mode == TEST_RUN_MODE
    assert parsed.func.__name__ == "run_cli"


def test_that_ensemble_experiment_with_realizations_can_be_selected_via_cli():
    # WHEN I select to run an ensemble experiment with specified realizations
    parsed = ert_parser(
        None,
        [ENSEMBLE_EXPERIMENT_MODE, "--realizations", "1-4,7,8", "config.ert"],
    )

    # THEN I expect ert to run those realizations in the cli
    assert parsed.mode == ENSEMBLE_EXPERIMENT_MODE
    assert parsed.realizations == "1-4,7,8"
    assert parsed.func.__name__ == "run_cli"


def test_that_ensemble_experiment_with_current_case_can_be_selected_via_li():
    # WHEN I select to run an ensemble experiment with specified case
    parsed = ert_parser(
        None,
        [ENSEMBLE_EXPERIMENT_MODE, "--current-case", "test_case", "config.ert"],
    )

    # THEN I expect it to run that case in the cli
    assert parsed.mode == ENSEMBLE_EXPERIMENT_MODE
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_that_incorrect_realizations_give_good_error_message(capsys):
    # WHEN I write an incorrect realizations string
    with pytest.raises(SystemExit):
        ert_parser(
            None,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--realizations",
                "1~4,7,",
                "config.ert",
            ],
        )

    # THEN I expect a reasonable error message
    assert "error: argument --realizations" in capsys.readouterr().err


def test_that_ensemble_smoother_can_be_selected_via_cli():
    # WHEN I run ensemble smoother with selected cases via the cli
    parsed = ert_parser(
        None,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--target-case",
            "some_case",
            "--current-case",
            "test_case",
            "config.ert",
        ],
    )

    # THEN I expect ensemble smoother to be run with those cases in the cli
    assert parsed.mode == ENSEMBLE_SMOOTHER_MODE
    assert parsed.target_case == "some_case"
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_that_ensemble_smoother_without_target_case_gives_good_error_message(capsys):
    # WHEN ensemble smoother is selected without target case
    with pytest.raises(SystemExit):
        ert_parser(None, [ENSEMBLE_SMOOTHER_MODE, "config.ert"])

    # THEN I expect to get a good error message
    assert "following arguments are required: --target-case" in capsys.readouterr().err


def test_that_iterative_ensemble_smoother_can_be_selected_via_cli():
    # WHEN I run iterative ensemble smoother with selected cases via the cli
    parsed = ert_parser(
        None,
        [
            ITERATIVE_ENSEMBLE_SMOOTHER_MODE,
            "--target-case",
            "some_case_%d",
            "--current-case",
            "test_case",
            "config.ert",
        ],
    )
    # THEN I expect iterative ensemble smoother to run with those cases in the cli
    assert parsed.mode == ITERATIVE_ENSEMBLE_SMOOTHER_MODE
    assert parsed.target_case == "some_case_%d"
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_that_ies_without_target_case_gives_good_error_message(capsys):
    # WHEN I select iterative ensemble smoother without target case
    with pytest.raises(SystemExit):
        ert_parser(None, [ITERATIVE_ENSEMBLE_SMOOTHER_MODE, "config.ert"])

    # THEN I expect to get a good error message
    assert "following arguments are required: --target-case" in capsys.readouterr().err


def test_that_selecting_es_mda_can_be_selected_from_the_commandline():
    # WHEN I run es_mda via the cli with my weights
    parsed = ert_parser(
        None,
        [
            ES_MDA_MODE,
            "--target-case",
            "some_case%d",
            "--current-case",
            "test_case",
            "--realizations",
            "1-10",
            "--weights",
            "1, 2, 4",
            "--start-iteration",
            "1",
            "config.ert",
        ],
    )
    # THEN I expect ES_MDA to run in the cli with my weights
    assert parsed.mode == ES_MDA_MODE
    assert parsed.target_case == "some_case%d"
    assert parsed.realizations == "1-10"
    assert parsed.weights == "1, 2, 4"
    assert parsed.start_iteration == "1"
    assert parsed.current_case == "test_case"
    assert parsed.func.__name__ == "run_cli"


def test_that_es_mda_has_correct_default_weights():
    # WHEN I run es_mda via the cli without weights
    parsed = ert_parser(None, [ES_MDA_MODE, "config.ert"])

    # THEN I expect the default weights to be selected
    assert parsed.mode == ES_MDA_MODE
    assert parsed.weights == "4, 2, 1"
    assert parsed.func.__name__ == "run_cli"


def test_that_a_workflow_can_be_run_via_cli():
    # WHEN I select a workflow to run via the cli
    parsed = ert_parser(None, [WORKFLOW_MODE, "workflow_name", "config.ert"])

    # THEN I expect that workflow to be run in the cli
    assert parsed.mode == WORKFLOW_MODE
    assert parsed.name == "workflow_name"
    assert parsed.func.__name__ == "run_cli"


@pytest.mark.parametrize(
    "port_input,expected_range",
    [("10-20", range(10, 21)), ("0-65535", range(0, 65536)), ("1-1", range(1, 2))],
)
def test_that_port_range_can_be_selected_via_cli(port_input, expected_range):
    # WHEN I select a port-range is selected
    parsed = ert_parser(
        None,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "--port-range",
            port_input,
            "config.ert",
        ],
    )
    # THEN ert runs with the selected port range
    assert parsed.port_range == expected_range


@pytest.mark.parametrize(
    "port_input",
    [("20-10"), ("65535"), ("1--31"), ("0-65536")],
)
def test_that_invalid_port_range_gives_good_error_message(port_input, capsys):

    # WHEN I select an incorrect port range
    with pytest.raises(SystemExit):
        ert_parser(
            None,
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "--port-range",
                port_input,
                "config.ert",
            ],
        )

    # THEN I expect to get a good error message
    assert "error: argument --port-range: " in capsys.readouterr().err


def test_that_no_port_range():
    # WHEN I don't specify a port range
    parsed = ert_parser(
        None,
        [
            ENSEMBLE_EXPERIMENT_MODE,
            "config.ert",
        ],
    )
    # THEN the default port range is used
    assert parsed.port_range == None


def test_that_when_target_case_equal_current_case_ert_errors(tmpdir):
    with tmpdir.as_cwd():
        # WHEN I erronously select target-case equal to current_case
        Path("config.ert").write_text("NUM_REALIZATIONS 1\n")
        parsed = ert_parser(
            None,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "test_case",
                "--target-case",
                "test_case",
                "config.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        # THEN I expect to get a good error message
        with pytest.raises(ErtCliError, match="They were both: test_case"):
            run_cli(parsed)


@pytest.mark.parametrize(
    "flag_val,expected_value", [("--verbose", True), (None, False)]
)
def test_verbose_flag(flag_val, expected_value):
    # WHEN I specify verbose via commandline
    args = [TEST_RUN_MODE, " config.ert"]
    if flag_val:
        args.append(flag_val)
    parsed = ert_parser(None, args)

    # THEN ert runs with verbose
    assert parsed.verbose == expected_value


def test_that_cli_can_display_version(capsys):
    # WHEN I ask for the ert version via cli
    try:
        ert_parser(None, ["--version"])
    except SystemExit as e:
        assert e.code == 0

    # THEN the version is displayed in cli
    ert_version, _ = capsys.readouterr()
    ert_version = ert_version.rstrip("\n")
    assert Version(ert_version)
