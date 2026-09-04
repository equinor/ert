import argparse
import re
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from fmudesign import (
    DesignMatrix,
    fmudesignrunner,
)
from fmudesign.fmudesignrunner import (
    Example,
    get_parser,
    subcommand_init,
    subcommand_run,
)


def test_that_given_unknown_command_run_is_inserted_at_position_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fmudesign", "foo"])

    assert sys.argv[1] == "foo"

    mock_args = MagicMock(func=lambda _: None)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)
    fmudesignrunner.main()

    assert sys.argv == ["fmudesign", "run", "foo"]
    assert sys.argv[1] == "run"


def test_that_missing_func_raises_system_exit_with_descriptive_error(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["fmudesign", "foo"])

    assert sys.argv[1] == "foo"

    mock_args = Namespace()
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)
    with pytest.raises(SystemExit) as sys_exit:
        fmudesignrunner.main()
    assert sys_exit.value.code == 0

    stdout = capsys.readouterr().out
    assert "usage: fmudesign [-v] [-h] {run,init}" in stdout


def test_that_main_catches_and_formats_validation_errors(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fmudesign", "foo"])

    def validation_error(_):
        class Foo(BaseModel):
            x: int

        Foo(x="foo")

    mock_args = MagicMock(func=validation_error)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)

    with pytest.raises(SystemExit) as sys_exit:
        fmudesignrunner.main()
    assert sys_exit.value.code == 1

    stdout = capsys.readouterr().out
    assert (
        "Validation error for 'x': Input should be a valid integer, "
        "unable to parse string as an integer, was 'foo'"
    ) in stdout


def test_that_main_catches_generic_exception_and_prints_traceback_with_message(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["fmudesign", "foo"])

    def validation_error(_):
        raise Exception("foo")

    mock_args = MagicMock(func=validation_error)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)

    with pytest.raises(SystemExit) as sys_exit:
        fmudesignrunner.main()
    assert sys_exit.value.code == 1

    captured = capsys.readouterr()
    assert "Traceback (most recent call last):" in captured.err
    assert "Exception: foo" in captured.err
    assert (
        "fmudesign failed. Read the error message above and fix the input file."
        in captured.out
    )


def _create_run_args(
    general_sheet="general_input",
    design_sheet="designinput",
    default_sheet="defaultvalues",
    config_file_name="foo.xlsx",
    destination_file_name="bar.xlsx",
):
    Path(config_file_name).touch()

    return Namespace(
        config=config_file_name,
        destination=destination_file_name,
        general_input=general_sheet,
        designinput=design_sheet,
        defaultvalues=default_sheet,
        verbose=1,
    )


def _mock_design_generation(monkeypatch):

    def noop(*_args, **_kwargs):
        pass

    monkeypatch.setattr(fmudesignrunner, "excel_to_dict", noop)
    monkeypatch.setattr(DesignMatrix, "generate", noop)
    monkeypatch.setattr(DesignMatrix, "to_xlsx", noop)


def test_that_subcommand_run_informs_when_default_sheet_has_changed(
    monkeypatch, use_tmpdir, capsys
):
    _, subparsers = get_parser()
    parser_run = subparsers.choices["run"]

    other_sheet = "foo"
    args = _create_run_args(
        general_sheet=other_sheet,
        default_sheet=other_sheet,
        design_sheet=other_sheet,
    )
    _mock_design_generation(monkeypatch)

    subcommand_run(args, parser_run)

    stdout = capsys.readouterr().out
    for sheet in ["general_input", "designinput", "defaultvalues"]:
        assert f"Worksheet changed from default: '{sheet}' -> '{other_sheet}'" in stdout


def test_that_subcommand_run_raises_when_config_does_not_exist(use_tmpdir):
    _, subparsers = get_parser()
    parser_run = subparsers.choices["run"]
    missing_file = "does_not_exist"
    args = _create_run_args(config_file_name=missing_file)
    Path(missing_file).unlink()
    with pytest.raises(OSError, match=f"Input file {missing_file} does not exist"):
        subcommand_run(args, parser_run)


def test_that_subcommand_run_raises_when_config_is_equal_to_destination(use_tmpdir):
    _, subparsers = get_parser()
    parser_run = subparsers.choices["run"]
    duplicate_name = "duplicate_name"
    Path(duplicate_name).touch()
    args = _create_run_args(
        config_file_name=duplicate_name, destination_file_name=duplicate_name
    )
    with pytest.raises(
        OSError, match=rf'Identical name "{duplicate_name}".*input.*output file'
    ):
        subcommand_run(args, parser_run)


def test_that_subcommand_init_raises_if_example_doesnt_exist(monkeypatch):
    _, subparsers = get_parser()
    parser_init = subparsers.choices["init"]
    monkeypatch.setattr(
        fmudesignrunner,
        "EXAMPLES",
        [Example("foo.xlsx", description="")],
    )
    config_file = "foo.xlsx"
    with pytest.raises(
        AssertionError, match=f"Example file '{config_file}' does not exist"
    ):
        subcommand_init(Namespace(file="config.xlsx"), parser_init)


def test_that_subcommand_init_raises_if_config_is_not_provided(capsys):
    _, subparsers = get_parser()
    parser_init = subparsers.choices["init"]
    with pytest.raises(SystemExit) as sys_exit:
        subcommand_init(Namespace(file=""), parser_init)
    assert sys_exit.value.code == 0

    stdout = capsys.readouterr().out
    assert "example usage:" in stdout
    assert re.search(r"fmudesign init .*\.xlsx", stdout)


def test_that_subcommand_init_raises_if_config_does_not_exist_among_examples(capsys):
    _, subparsers = get_parser()
    parser_init = subparsers.choices["init"]
    with pytest.raises(SystemExit) as sys_exit:
        subcommand_init(Namespace(file="config.xlsx"), parser_init)
    assert sys_exit.value.code == 1

    stdout = capsys.readouterr().out
    assert "Error on 'config.xlsx'. Not found among:" in stdout


def test_that_subcommand_init_raises_if_file_already_exists(use_tmpdir, capsys):
    _, subparsers = get_parser()
    parser_init = subparsers.choices["init"]
    existing_file = "fmudesign_ex_montecarlo.xlsx"
    Path(existing_file).touch()

    with pytest.raises(SystemExit) as sys_exit:
        subcommand_init(Namespace(file=existing_file), parser_init)
    assert sys_exit.value.code == 1

    stdout = capsys.readouterr().out
    assert f"Error on '{existing_file}'. Already exists." in stdout


def test_that_subcommand_init_creates_example_file(use_tmpdir, capsys):
    _, subparsers = get_parser()
    parser_init = subparsers.choices["init"]
    example_file = "fmudesign_ex_montecarlo.xlsx"

    with pytest.raises(SystemExit) as sys_exit:
        subcommand_init(Namespace(file=example_file), parser_init)
    assert sys_exit.value.code == 0

    stdout = capsys.readouterr().out
    assert Path(example_file).is_file()
    assert f"Created file {example_file!r}." in stdout


def test_that_subcommand_init_creates_auxiliary_files(use_tmpdir, capsys):
    _, subparsers = get_parser()
    parser_init = subparsers.choices["init"]
    example_file = "ex2_correlations.xlsx"
    auxiliary_file = "ex2_doe1.xlsx"

    with pytest.raises(SystemExit) as sys_exit:
        subcommand_init(Namespace(file=example_file), parser_init)
    assert sys_exit.value.code == 0

    stdout = capsys.readouterr().out
    assert Path(example_file).is_file()
    assert Path(auxiliary_file).is_file()
    assert f"Created file {example_file!r}." in stdout
    assert f"Created auxiliary file {auxiliary_file!r}." in stdout
