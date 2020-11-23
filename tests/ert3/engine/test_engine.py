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
    pathlib.Path(os.path.dirname(__file__)) / ".." / ".." / ".." / "examples"
)
_POLY_WORKSPACE_NAME = "polynomial"
_POLY_WORKSPACE = _EXAMPLES_ROOT / _POLY_WORKSPACE_NAME


def test_engine_run_no_workspace(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    err_msg = "????????"
    with pytest.raises(SystemError, match=err_msg):
        ert3.engine.run(os.getcwd(), "not-an-experiment")


def test_engine_run_unknown_experiment(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()
    
    ert3.workspace.init(workspace)
    err_msg = "not-an-experiment is not an experiment within the workspace"
    with pytest.raises(ValueError, match=err_msg):
        ert3.engine.run(os.getcwd(), "not-an-experiment")


def test_engine_run(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)

    ert3.workspace.init(workspace)
    ert3.engine.run(workspace, "evaluation")
    output_data = ert3.storage.get_output_data(workspace, "evaluation")
    
    assert len(output_data) == 1000
    for entry in output_data:
        assert ["polynomial_result"] == list(entry.keys())
        assert 10 == len(entry["polynomial_result"])

    with pytest.raises(ValueError, match="TODO: What is this?"):
        ert3.engine.run(workspace, "evaluation")


def test_export_unknown_experiment(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()
    
    ert3.workspace.init(workspace)
    with pytest.raises(SystemExit, match="Something todo"):
        ert3.engine.export(os.getcwd(), "not-an-experiment")


def test_export_not_run(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()

    ert3.workspace.init(workspace)
    ert3.engine.export(workspace, "evaluation")


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


def test_export_polynomial_evaluation(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)

    ert3.workspace.init(workspace)
    ert3.engine.run(workspace, "evaluation")
    ert3.engine.export(workspace, "evaluation")
    _assert_export(workspace, "evaluation")
