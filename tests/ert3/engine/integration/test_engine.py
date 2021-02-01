import pathlib
import os
import shutil
import numbers

import json
import yaml

import pytest

import ert3
from tests.ert3.engine.integration.conftest import (
    assert_sensitivity_oat_export,
    assert_export,
    assert_distribution,
)

_EXAMPLES_ROOT = (
    pathlib.Path(os.path.dirname(__file__)) / ".." / ".." / ".." / ".." / "examples"
)


@pytest.fixture()
def base_ensemble_dict():
    yield {
        "size": 10,
        "input": [{"source": "stochastic.coefficients", "record": "coefficients"}],
        "forward_model": {"driver": "local", "stages": ["evaluate_polynomial"]},
    }


@pytest.fixture()
def ensemble(base_ensemble_dict):
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def sensitivity_ensemble(base_ensemble_dict):
    base_ensemble_dict.pop("size")
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def uniform_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "stochastic.uniform_coefficients"
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def presampled_uniform_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "storage.uniform_coefficients0"
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def presampled_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "storage.coefficients0"
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def doe_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "storage.designed_coefficients"
    base_ensemble_dict["size"] = 1000
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def big_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "storage.coefficients0"
    base_ensemble_dict["size"] = 1000
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def presampled_big_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "storage.uniform_coefficients0"
    base_ensemble_dict["size"] = 1000
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def stages_config():
    config_list = [
        {
            "name": "evaluate_polynomial",
            "input": [{"record": "coefficients", "location": "coefficients.json"}],
            "output": [{"record": "polynomial_output", "location": "output.json"}],
            "script": ["poly --coefficients coefficients.json --output output.json"],
            "transportable_commands": [
                {
                    "name": "poly",
                    "location": "poly.py",
                }
            ],
        }
    ]
    shutil.copy2(_EXAMPLES_ROOT / "polynomial" / "poly.py", "poly.py")
    yield ert3.config.load_stages_config(config_list)


@pytest.fixture()
def evaluation_experiment_config():
    raw_config = {"type": "evaluation"}
    yield ert3.config.load_experiment_config(raw_config)


@pytest.fixture()
def sensitivity_experiment_config():
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time"}
    yield ert3.config.load_experiment_config(raw_config)


@pytest.fixture()
def gaussian_parameters_file():
    content = [
        {
            "name": "coefficients",
            "type": "stochastic",
            "distribution": {"type": "gaussian", "input": {"mean": 0, "std": 1}},
            "variables": ["a", "b", "c"],
        },
    ]
    with open("parameters.yml", "w") as fout:
        yaml.dump(content, fout)


@pytest.fixture()
def uniform_parameters_file():
    content = [
        {
            "name": "uniform_coefficients",
            "type": "stochastic",
            "distribution": {
                "type": "uniform",
                "input": {"lower_bound": 0, "upper_bound": 1},
            },
            "variables": ["a", "b", "c"],
        },
    ]
    with open("parameters.yml", "w") as fout:
        yaml.dump(content, fout)


@pytest.fixture()
def workspace(tmpdir):
    workspace = tmpdir / "polynomial"
    workspace.mkdir()
    workspace.chdir()
    ert3.workspace.initialize(workspace)
    yield workspace


def test_run_once_polynomial_evaluation(
    workspace,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_file,
):
    ert3.engine.run(
        ensemble, stages_config, evaluation_experiment_config, workspace, "evaluation"
    )
    with pytest.raises(ValueError, match="Experiment evaluation have been carried out"):
        ert3.engine.run(
            ensemble,
            stages_config,
            evaluation_experiment_config,
            workspace,
            "evaluation",
        )


def test_export_not_run(workspace):
    (workspace / "evaluation").mkdir()
    with pytest.raises(ValueError, match="Cannot export experiment"):
        ert3.engine.export(pathlib.Path(), "evaluation")


def test_export_polynomial_evaluation(
    workspace,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_file,
):
    (workspace / "evaluation").mkdir()
    ert3.engine.run(
        ensemble, stages_config, evaluation_experiment_config, workspace, "evaluation"
    )
    ert3.engine.export(workspace, "evaluation")

    assert_export(workspace, "evaluation", ensemble, stages_config)


def test_export_uniform_polynomial_evaluation(
    workspace,
    uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_file,
):
    (workspace / "uniform_evaluation").mkdir()
    ert3.engine.run(
        uniform_ensemble,
        stages_config,
        evaluation_experiment_config,
        workspace,
        "uniform_evaluation",
    )
    ert3.engine.export(workspace, "uniform_evaluation")

    assert_export(workspace, "uniform_evaluation", uniform_ensemble, stages_config)


def test_gaussian_distribution(
    workspace, big_ensemble, stages_config, gaussian_parameters_file
):
    ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 1000)

    coefficients = ert3.storage.get_variables(workspace, "coefficients0")
    assert 1000 == len(coefficients)

    assert_distribution(
        workspace, big_ensemble, stages_config, "gaussian", coefficients
    )


def test_uniform_distribution(
    workspace, presampled_big_ensemble, stages_config, uniform_parameters_file
):
    ert3.engine.sample_record(
        workspace, "uniform_coefficients", "uniform_coefficients0", 1000
    )

    coefficients = ert3.storage.get_variables(workspace, "uniform_coefficients0")
    assert 1000 == len(coefficients)

    assert_distribution(
        workspace, presampled_big_ensemble, stages_config, "uniform", coefficients
    )


def test_run_presampled(
    workspace,
    presampled_ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_file,
):
    (workspace / "presampled_evaluation").mkdir()
    ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 10)

    coeff0 = ert3.storage.get_variables(workspace, "coefficients0")
    assert 10 == len(coeff0)
    for real_coeff in coeff0:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.keys())
        for val in real_coeff.values():
            assert isinstance(val, float)

    ert3.engine.run(
        presampled_ensemble,
        stages_config,
        evaluation_experiment_config,
        workspace,
        "presampled_evaluation",
    )
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


def test_run_uniform_presampled(
    workspace,
    presampled_uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_file,
):
    (workspace / "presampled_uniform_evaluation").mkdir()
    ert3.engine.sample_record(
        workspace, "uniform_coefficients", "uniform_coefficients0", 10
    )

    uniform_coeff0 = ert3.storage.get_variables(workspace, "uniform_coefficients0")
    assert 10 == len(uniform_coeff0)
    for real_coeff in uniform_coeff0:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.keys())
        for val in real_coeff.values():
            assert isinstance(val, float)

    ert3.engine.run(
        presampled_uniform_ensemble,
        stages_config,
        evaluation_experiment_config,
        workspace,
        "presampled_uniform_evaluation",
    )
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


def test_sample_unknown_parameter_group(workspace, uniform_parameters_file):
    with pytest.raises(ValueError, match="No parameter group found named: coeffs"):
        ert3.engine.sample_record(workspace, "coeffs", "coefficients0", 100)


def test_sample_unknown_distribution(workspace, gaussian_parameters_file):
    with open(workspace / "parameters.yml") as f:
        parameters = yaml.safe_load(f)
    parameters[0]["distribution"]["type"] = "double-hyper-exp"
    with open(workspace / "parameters.yml", "w") as f:
        yaml.dump(parameters, f)

    with pytest.raises(ValueError, match="Unknown distribution type: double-hyper-exp"):
        ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 100)


def test_record_load_and_run(
    workspace, doe_ensemble, stages_config, evaluation_experiment_config
):
    pathlib.Path("doe").mkdir()
    coeffs_file = _EXAMPLES_ROOT / "polynomial" / "doe" / "coefficients_record.json"
    shutil.copy(coeffs_file, "doe")
    record_file = (pathlib.Path("doe") / "coefficients_record.json").open("r")
    ert3.engine.load_record(workspace, "designed_coefficients", record_file)

    designed_coeff = ert3.storage.get_variables(workspace, "designed_coefficients")
    assert 1000 == len(designed_coeff)
    for real_coeff in designed_coeff:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.keys())
        for val in real_coeff.values():
            assert isinstance(val, numbers.Number)

    ert3.engine.run(
        doe_ensemble, stages_config, evaluation_experiment_config, workspace, "doe"
    )
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


def test_record_load_twice(workspace, ensemble, stages_config):
    pathlib.Path("doe").mkdir()
    coeffs_file = _EXAMPLES_ROOT / "polynomial" / "doe" / "coefficients_record.json"
    shutil.copy(coeffs_file, "doe")
    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    ert3.engine.load_record(workspace, "designed_coefficients", record_file)
    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    with pytest.raises(KeyError):
        ert3.engine.load_record(workspace, "designed_coefficients", record_file)


def test_sensitivity_run_and_export(
    workspace,
    sensitivity_ensemble,
    stages_config,
    sensitivity_experiment_config,
    gaussian_parameters_file,
):
    pathlib.Path("sensitivity").mkdir()
    ert3.engine.run(
        sensitivity_ensemble,
        stages_config,
        sensitivity_experiment_config,
        workspace,
        "sensitivity",
    )
    ert3.engine.export(workspace, "sensitivity")
    assert_sensitivity_oat_export(
        workspace, "sensitivity", sensitivity_ensemble, stages_config
    )
