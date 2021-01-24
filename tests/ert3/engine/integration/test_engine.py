import pathlib
import os
import shutil
import numbers

import json
import yaml

import pytest

import ert3
from tests.ert3.engine.integration.conftest import assert_distribution

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


def test_record_load_twice(workspace, ensemble, stages_config):
    pathlib.Path("doe").mkdir()
    coeffs_file = _EXAMPLES_ROOT / "polynomial" / "doe" / "coefficients_record.json"
    shutil.copy(coeffs_file, "doe")
    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    ert3.engine.load_record(workspace, "designed_coefficients", record_file)
    record_file = (workspace / "doe" / "coefficients_record.json").open("r")
    with pytest.raises(KeyError):
        ert3.engine.load_record(workspace, "designed_coefficients", record_file)
