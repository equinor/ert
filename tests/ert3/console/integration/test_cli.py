import ert3

import json
import numpy as np
import os
import pathlib
import pytest
import shutil
import sys
import unittest
import yaml

_EXAMPLES_ROOT = (
    pathlib.Path(os.path.dirname(__file__)) / ".." / ".." / ".." / ".." / "examples"
)
_POLY_WORKSPACE_NAME = "polynomial"
_POLY_WORKSPACE = _EXAMPLES_ROOT / _POLY_WORKSPACE_NAME


@pytest.mark.parametrize(
    "args",
    [
        ["ert3", "run", "something"],
        ["ert3", "export", "something"],
    ],
)
def test_cli_no_init(tmpdir, args):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(SystemExit) as error:
            ert3.console.main()
        assert "Not inside an ERT workspace" in str(error.value)


def test_cli_no_args(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()


def test_cli_init(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()


def test_cli_init_twice(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
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
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    subfolder = tmpdir / _POLY_WORKSPACE_NAME / "subfolder"
    subfolder.mkdir()
    subfolder.chdir()

    with unittest.mock.patch.object(sys, "argv", args):
        with pytest.raises(SystemExit) as error:
            ert3.console.main()
        assert "Already inside an ERT workspace" in str(error.value)


def test_cli_run_invalid_experiment(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
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


def test_cli_run_once_polynomial_evaluation(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
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
        assert "Experiment evaluation have been carried out" in str(error.value)


def test_cli_run_polynomial_evaluation(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    args = ["ert3", "init"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()

    args = ["ert3", "run", "evaluation"]
    with unittest.mock.patch.object(sys, "argv", args):
        ert3.console.main()


def test_cli_export_not_run(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
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


def _load_experiment_config(workspace, experiment_name):
    config = {}
    with open(workspace / experiment_name / "ensemble.yml") as f:
        config["ensemble"] = yaml.safe_load(f)
    with open(workspace / "stages.yml") as f:
        config["stages"] = yaml.safe_load(f)
    with open(workspace / "parameters.yml") as f:
        config["parameters"] = yaml.safe_load(f)
    return config


def _assert_ensemble_size(config, export_data):
    assert len(export_data) == config["ensemble"]["size"]


def _assert_input_records(config, export_data):
    input_records = {}
    for input_data in config["ensemble"]["input"]:
        record = input_data["record"]
        source = input_data["source"]

        for p in config["parameters"]:
            if p["type"] + "." + p["name"] == source:
                parameter = p
                break

        input_records[record] = parameter["variables"]

    for realisation in export_data:
        assert sorted(input_records.keys()) == sorted(realisation["input"].keys())
        for record_name in input_records.keys():
            input_variables = sorted(input_records[record_name])
            realisation_variables = sorted(realisation["input"][record_name].keys())
            assert input_variables == realisation_variables


def _assert_output_records(config, export_data):
    output_records = []
    for forward_stage in config["ensemble"]["forward_model"]["stages"]:
        for stage in config["stages"]:
            if stage["name"] == forward_stage:
                output_records += [
                    output_data["record"] for output_data in stage["output"]
                ]
    for realisation in export_data:
        assert sorted(output_records) == sorted(realisation["output"].keys())


def _assert_poly_output(config, export_data):
    for realisation in export_data:
        coeff = realisation["input"]["coefficients"]
        poly_out = realisation["output"]["polynomial_output"]

        assert 10 == len(poly_out)
        for x, y in zip(range(10), poly_out):
            assert coeff["a"] * x ** 2 + coeff["b"] * x + coeff["c"] == pytest.approx(y)


def _assert_parameter_statistics(config, export_data):
    for input_data in config["ensemble"]["input"]:
        record = input_data["record"]
        source = input_data["source"]

        for p in config["parameters"]:
            if p["type"] + "." + p["name"] == source:
                parameter = p
                break

        assert parameter["distribution"]["type"] == "gaussian"
        mean = parameter["distribution"]["input"]["mean"]
        std = parameter["distribution"]["input"]["std"]

        for variable in parameter["variables"]:
            values = np.array(
                [realisation["input"][record][variable] for realisation in export_data]
            )
            assert mean == pytest.approx(sum(values) / len(values), abs=0.1)
            assert std == pytest.approx(np.std(values), abs=0.1)


def _assert_export(workspace, experiment_name):
    with open(workspace / experiment_name / "data.json") as f:
        export_data = json.load(f)

    config = _load_experiment_config(workspace, experiment_name)
    _assert_ensemble_size(config, export_data)
    _assert_input_records(config, export_data)
    _assert_output_records(config, export_data)

    # Note: This test assumes the forward model in the setup indeed
    # evaluates a * x^2 + b * x + c. If not, this will fail miserably!
    _assert_poly_output(config, export_data)

    # Note: This might fail (but with rather low probability) as it computes
    # the mean and std of the sampled parameter values and compares it to the
    # theoretical distribution.
    _assert_parameter_statistics(config, export_data)


def test_cli_export_polynomial_evaluation(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
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

    _assert_export(workspace, "evaluation")
