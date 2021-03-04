import pathlib
import numbers
import json
import yaml
import pytest

import ert3

from tests.ert3.conftest import (
    assert_sensitivity_oat_export,
    assert_export,
    assert_distribution,
)


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
    base_ensemble_dict["size"] = 10
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
def designed_coeffs_record_file(workspace):
    doe_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "doe"
    doe_dir.ensure(dir=True)
    coeffs = [{"a": x, "b": x, "c": x} for x in range(10)]
    with open(doe_dir / "coefficients_record.json", "w") as f:
        json.dump(coeffs, f)
    yield doe_dir / "coefficients_record.json"


@pytest.mark.requires_ert_storage
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


@pytest.mark.requires_ert_storage
def test_export_not_run(workspace):
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "evaluation").ensure(dir=True)
    with pytest.raises(ValueError, match="Cannot export experiment"):
        ert3.engine.export(pathlib.Path(), "evaluation")


@pytest.mark.requires_ert_storage
def test_export_polynomial_evaluation(
    workspace,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_file,
):
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "evaluation").ensure(dir=True)
    ert3.engine.run(
        ensemble, stages_config, evaluation_experiment_config, workspace, "evaluation"
    )
    ert3.engine.export(workspace, "evaluation")

    assert_export(workspace, "evaluation", ensemble, stages_config)


@pytest.mark.requires_ert_storage
def test_export_uniform_polynomial_evaluation(
    workspace,
    uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_file,
):
    uni_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "uniform_evaluation"
    uni_dir.ensure(dir=True)
    ert3.engine.run(
        uniform_ensemble,
        stages_config,
        evaluation_experiment_config,
        workspace,
        "uniform_evaluation",
    )
    ert3.engine.export(workspace, "uniform_evaluation")

    assert_export(workspace, "uniform_evaluation", uniform_ensemble, stages_config)


@pytest.mark.requires_ert_storage
def test_gaussian_distribution(
    workspace, big_ensemble, stages_config, gaussian_parameters_file
):
    ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 1000)

    coefficients = ert3.storage.get_ensemble_record(
        workspace=workspace, record_name="coefficients0"
    )
    assert 1000 == coefficients.ensemble_size

    assert_distribution(
        workspace, big_ensemble, stages_config, "gaussian", coefficients
    )


@pytest.mark.requires_ert_storage
def test_uniform_distribution(
    workspace,
    presampled_big_ensemble,
    stages_config,
    uniform_parameters_file,
):
    ert3.engine.sample_record(
        workspace, "uniform_coefficients", "uniform_coefficients0", 1000
    )

    coefficients = ert3.storage.get_ensemble_record(
        workspace=workspace, record_name="uniform_coefficients0"
    )
    assert 1000 == coefficients.ensemble_size

    assert_distribution(
        workspace, presampled_big_ensemble, stages_config, "uniform", coefficients
    )


@pytest.mark.requires_ert_storage
def test_run_presampled(
    workspace,
    presampled_ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_file,
):
    presampled_dir = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / "presampled_evaluation"
    )
    presampled_dir.ensure(dir=True)
    ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 10)

    coeff0 = ert3.storage.get_ensemble_record(
        workspace=workspace, record_name="coefficients0"
    )
    assert 10 == coeff0.ensemble_size
    for real_coeff in coeff0.records:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
        for idx in real_coeff.index:
            assert isinstance(real_coeff.data[idx], float)

    ert3.engine.run(
        presampled_ensemble,
        stages_config,
        evaluation_experiment_config,
        workspace,
        "presampled_evaluation",
    )
    ert3.engine.export(workspace, "presampled_evaluation")

    export_file = presampled_dir / "data.json"
    with open(export_file) as f:
        export_data = json.load(f)

    assert coeff0.ensemble_size == len(export_data)
    for coeff, real in zip(coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.requires_ert_storage
def test_run_uniform_presampled(
    workspace,
    presampled_uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_file,
):
    presampled_dir = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / "presampled_uniform_evaluation"
    )
    presampled_dir.ensure(dir=True)
    ert3.engine.sample_record(
        workspace, "uniform_coefficients", "uniform_coefficients0", 10
    )

    uniform_coeff0 = ert3.storage.get_ensemble_record(
        workspace=workspace, record_name="uniform_coefficients0"
    )
    assert 10 == uniform_coeff0.ensemble_size
    for real_coeff in uniform_coeff0.records:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
        for idx in real_coeff.index:
            assert isinstance(real_coeff.data[idx], float)

    ert3.engine.run(
        presampled_uniform_ensemble,
        stages_config,
        evaluation_experiment_config,
        workspace,
        "presampled_uniform_evaluation",
    )
    ert3.engine.export(workspace, "presampled_uniform_evaluation")

    with open(presampled_dir / "data.json") as f:
        export_data = json.load(f)

    assert uniform_coeff0.ensemble_size == len(export_data)
    for coeff, real in zip(uniform_coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.requires_ert_storage
def test_sample_unknown_parameter_group(workspace, uniform_parameters_file):
    with pytest.raises(ValueError, match="No parameter group found named: coeffs"):
        ert3.engine.sample_record(workspace, "coeffs", "coefficients0", 100)


@pytest.mark.requires_ert_storage
def test_sample_unknown_distribution(workspace, gaussian_parameters_file):
    with open(workspace / "parameters.yml") as f:
        parameters = yaml.safe_load(f)
    parameters[0]["distribution"]["type"] = "double-hyper-exp"
    with open(workspace / "parameters.yml", "w") as f:
        yaml.dump(parameters, f)

    with pytest.raises(ValueError, match="Unknown distribution type: double-hyper-exp"):
        ert3.engine.sample_record(workspace, "coefficients", "coefficients0", 100)


@pytest.mark.requires_ert_storage
def test_record_load_and_run(
    workspace,
    doe_ensemble,
    stages_config,
    evaluation_experiment_config,
    designed_coeffs_record_file,
):

    with open(designed_coeffs_record_file) as rs:
        ert3.engine.load_record(workspace, "designed_coefficients", rs)
    designed_coeff = ert3.storage.get_ensemble_record(
        workspace=workspace, record_name="designed_coefficients"
    )
    assert 10 == designed_coeff.ensemble_size
    for real_coeff in designed_coeff.records:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
        for val in real_coeff.data.values():
            assert isinstance(val, numbers.Number)

    ert3.engine.run(
        doe_ensemble, stages_config, evaluation_experiment_config, workspace, "doe"
    )
    ert3.engine.export(workspace, "doe")

    export_file = workspace / ert3.workspace.EXPERIMENTS_BASE / "doe" / "data.json"
    with open(export_file) as f:
        export_data = json.load(f)

    assert designed_coeff.ensemble_size == len(export_data)
    for coeff, real in zip(designed_coeff.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.requires_ert_storage
def test_record_load_twice(
    workspace, ensemble, stages_config, designed_coeffs_record_file
):
    with open(designed_coeffs_record_file, "r") as record_file:
        ert3.engine.load_record(workspace, "designed_coefficients", record_file)
    with pytest.raises(KeyError):
        with open(designed_coeffs_record_file, "r") as record_file:
            ert3.engine.load_record(workspace, "designed_coefficients", record_file)


@pytest.mark.requires_ert_storage
def test_sensitivity_run_and_export(
    workspace,
    sensitivity_ensemble,
    stages_config,
    sensitivity_experiment_config,
    gaussian_parameters_file,
):
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "sensitivity").ensure(dir=True)
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
