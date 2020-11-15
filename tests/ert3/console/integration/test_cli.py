import ert3

import os
import pathlib
import pytest
import shutil
import sys
import unittest

_EXAMPLES_ROOT = pathlib.Path(os.path.dirname(__file__))/".."/".."/".."/".."/"examples"
_POLY_WORKSPACE_NAME = "polynomial"
_POLY_WORKSPACE = _EXAMPLES_ROOT/_POLY_WORKSPACE_NAME


@pytest.mark.parametrize(
    "args",
    [
        ["ert3", "run", "something"],
        ["ert3", "export", "something"],
    ]
)
def test_cli_no_init(tmpdir, args):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(SystemExit) as error:
            ert3.console.main()
        assert "Not inside an ERT workspace" in str(error.value)


def test_cli_no_args(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()


def test_cli_init(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()


def test_cli_init_twice(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(SystemExit) as error:
            ert3.console.main()
        assert "Already inside an ERT workspace" in str(error.value)


def test_cli_init_subfolder(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    subfolder = tmpdir/_POLY_WORKSPACE_NAME/"subfolder"
    subfolder.mkdir()
    subfolder.chdir()

    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(SystemExit) as error:
            ert3.console.main()
        assert "Already inside an ERT workspace" in str(error.value)


def test_cli_run_invalid_experiment(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "run", "this-is-not-an-experiment"]
    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(ValueError) as error:
            ert3.console.main()
        assert "this-is-not-an-experiment is not an experiment" in str(error.value)


def test_cli_run_polynomial_evaluation(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "run", "evaluation"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

        with pytest.raises(ValueError) as error:
            ert3.console.main()
        assert "Experiment evaluation have already been carried out" in str(error.value)


def test_cli_run_polynomial_evaluation(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "run", "evaluation"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()


def test_cli_export_not_run(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "export", "evaluation"]
    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(ValueError) as error:
            ert3.console.main()
        assert "Cannot export experiment" in str(error.value)


def test_cli_export_polynomial_evaluation(tmpdir):
    workspace = tmpdir/_POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "run", "evaluation"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "export", "evaluation"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    with open(workspace/"evaluation"/"data.csv") as f:
        data = pandas.read_csv(f)
