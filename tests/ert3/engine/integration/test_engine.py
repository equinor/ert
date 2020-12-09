import pathlib
import os
import shutil
import numbers

import json
import yaml

import pytest

import ert3
from ert3.console._console import _load_ensemble_config, _load_stages_config
from tests.ert3.engine.integration.conftest import assert_export, assert_distribution

_EXAMPLES_ROOT = (
    pathlib.Path(os.path.dirname(__file__)) / ".." / ".." / ".." / ".." / "examples"
)
_POLY_WORKSPACE_NAME = "polynomial"
_POLY_WORKSPACE = _EXAMPLES_ROOT / _POLY_WORKSPACE_NAME


def set_workspace(workspace, case_name="evaluation"):
    ert3.workspace.initialize(workspace)
    ensemble = _load_ensemble_config(workspace, case_name)
    stages_config = _load_stages_config(workspace)
    return workspace, ensemble, stages_config


@pytest.fixture()
def get_workspace(tmpdir):
    workspace = tmpdir / _POLY_WORKSPACE_NAME
    shutil.copytree(_POLY_WORKSPACE, workspace)
    workspace.chdir()
    yield workspace


@pytest.fixture()
def initialize_experiment(get_workspace):
    workspace = get_workspace
    yield set_workspace(workspace)


def test_run_once_polynomial_evaluation(tmpdir, initialize_experiment):
    workspace, ensemble, stages_config = initialize_experiment

    ert3.engine.run(ensemble, stages_config, workspace, "evaluation")
    with pytest.raises(ValueError, match="Experiment evaluation have been carried out"):
        ert3.engine.run(ensemble, stages_config, workspace, "evaluation")


@pytest.mark.usefixtures("initialize_experiment")
def test_export_not_run(tmpdir):
    with pytest.raises(ValueError, match="Cannot export experiment"):
        ert3.engine.export(pathlib.Path(), "evaluation")


def test_export_polynomial_evaluation(initialize_experiment):
    workspace, ensemble, stages_config = initialize_experiment

    ert3.engine.run(ensemble, stages_config, workspace, "evaluation")
    ert3.engine.export(workspace, "evaluation")

    assert_export(workspace, "evaluation")


def test_export_uniform_polynomial_evaluation(get_workspace):
    workspace, ensemble, stages_config = set_workspace(
        get_workspace, case_name="uniform_evaluation"
    )

    ert3.engine.run(ensemble, stages_config, workspace, "uniform_evaluation")
    ert3.engine.export(workspace, "uniform_evaluation")

    assert_export(workspace, "uniform_evaluation")


def test_gaussian_distribution(initialize_experiment):
    workspace, ensemble, stages_config = initialize_experiment
    ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 1000)

    coefficients = ert3.storage.get_variables(workspace, "coefficients0")
    assert 1000 == len(coefficients)

    assert_distribution(
        workspace, "presampled_evaluation_big", "gaussian", coefficients
    )


def test_uniform_distribution(initialize_experiment):
    workspace, ensemble, stages_config = initialize_experiment

    ert3.engine.sample_record(
        workspace, "uniform_coefficients", "uniform_coefficients0", 1000
    )

    coefficients = ert3.storage.get_variables(workspace, "uniform_coefficients0")
    assert 1000 == len(coefficients)

    assert_distribution(
        workspace, "presampled_uniform_evaluation_big", "uniform", coefficients
    )


def test_run_presampled(get_workspace):
    workspace, ensemble, stages_config = set_workspace(
        get_workspace, case_name="presampled_evaluation"
    )

    ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 10)

    coeff0 = ert3.storage.get_variables(workspace, "coefficients0")
    assert 10 == len(coeff0)
    for real_coeff in coeff0:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.keys())
        for val in real_coeff.values():
            assert isinstance(val, float)

    ert3.engine.run(ensemble, stages_config, workspace, "presampled_evaluation")
    ert3.engine.export(workspace, "presampled_evaluation")

    with open(workspace / "presampled_evaluation" / "data.json") as f:
        export_data = json.load(f)

    assert len(coeff0) == len(export_data)
    for coeff, real in zip(coeff0, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert coeff.keys() == export_coeff.keys()
        for key in coeff.keys():
            assert coeff[key] == export_coeff[key]


def test_run_uniform_presampled(get_workspace):
    workspace, ensemble, stages_config = set_workspace(
        get_workspace, case_name="presampled_uniform_evaluation"
    )

    ert3.engine.sample_record(
        workspace, "uniform_coefficients", "uniform_coefficients0", 10
    )

    uniform_coeff0 = ert3.storage.get_variables(workspace, "uniform_coefficients0")
    assert 10 == len(uniform_coeff0)
    for real_coeff in uniform_coeff0:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.keys())
        for val in real_coeff.values():
            assert isinstance(val, float)

    ert3.engine.run(ensemble, stages_config, workspace, "presampled_uniform_evaluation")
    ert3.engine.export(workspace, "presampled_uniform_evaluation")

    with open(workspace / "presampled_uniform_evaluation" / "data.json") as f:
        export_data = json.load(f)

    assert len(uniform_coeff0) == len(export_data)
    for coeff, real in zip(uniform_coeff0, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert coeff.keys() == export_coeff.keys()
        for key in coeff.keys():
            assert coeff[key] == export_coeff[key]


def test_sample_unknown_parameter_group(initialize_experiment):
    workspace, _, _ = initialize_experiment

    with pytest.raises(ValueError, match="No parameter group found named: coeffs"):
        ert3.engine.sample_record(workspace, "coeffs", "coefficients0", 100)


def test_sample_unknown_distribution(initialize_experiment):
    workspace, _, _ = initialize_experiment

    with open(workspace / "parameters.yml") as f:
        parameters = yaml.safe_load(f)
    parameters[0]["distribution"]["type"] = "double-hyper-exp"
    with open(workspace / "parameters.yml", "w") as f:
        yaml.dump(parameters, f)

    with pytest.raises(ValueError, match="Unknown distribution type: double-hyper-exp"):
        ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 100)


def test_record_load_and_run(get_workspace):
    workspace, ensemble, stages_config = set_workspace(get_workspace, case_name="doe")

    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    ert3.engine.load_record(workspace, "designed_coefficients", record_file)

    designed_coeff = ert3.storage.get_variables(workspace, "designed_coefficients")
    assert 1000 == len(designed_coeff)
    for real_coeff in designed_coeff:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.keys())
        for val in real_coeff.values():
            assert isinstance(val, numbers.Number)

    ert3.engine.run(ensemble, stages_config, workspace, "doe")
    ert3.engine.export(workspace, "doe")

    with open(workspace / "doe" / "data.json") as f:
        export_data = json.load(f)

    assert len(designed_coeff) == len(export_data)
    for coeff, real in zip(designed_coeff, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert coeff.keys() == export_coeff.keys()
        for key in coeff.keys():
            assert coeff[key] == export_coeff[key]


def test_record_load_twice(initialize_experiment):
    workspace, ensemble, stages_config = initialize_experiment
    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    ert3.engine.load_record(workspace, "designed_coefficients", record_file)
    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    with pytest.raises(KeyError):
        ert3.engine.load_record(workspace, "designed_coefficients", record_file)
