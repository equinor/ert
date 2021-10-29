import json
import pathlib

import pytest

from ert_shared.asyncio import get_event_loop
from integration_utils import (
    assert_distribution,
    assert_export,
    assert_sensitivity_export,
)

import ert
import ert3
from unittest.mock import patch


@pytest.fixture()
def sensitivity_ensemble(base_ensemble_dict):
    base_ensemble_dict.pop("size")
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def uniform_ensemble(base_ensemble_dict):
    base_ensemble_dict["input"][0]["source"] = "stochastic.uniform_coefficients"
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def x_uncertainty_ensemble(base_ensemble_dict):
    base_ensemble_dict["forward_model"]["stage"] = "evaluate_x_uncertainty_polynomial"
    base_ensemble_dict["input"].append(
        {"record": "x_uncertainties", "source": "stochastic.x_normals"}
    )
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
def partial_sensitivity_ensemble(base_ensemble_dict):
    base_ensemble_dict.pop("size")
    base_ensemble_dict["input"].append(
        {"record": "other_coefficients", "source": "storage.other_coefficients"}
    )
    base_ensemble_dict["output"].append({"record": "other_polynomial_output"})
    yield ert3.config.load_ensemble_config(base_ensemble_dict)


@pytest.fixture()
def evaluation_experiment_config():
    raw_config = {"type": "evaluation"}
    yield ert3.config.load_experiment_config(raw_config)


@pytest.fixture()
def sensitivity_oat_experiment_config():
    raw_config = {"type": "sensitivity", "algorithm": "one-at-a-time", "tail": 0.99}
    yield ert3.config.load_experiment_config(raw_config)


@pytest.fixture()
def sensitivity_fast_experiment_config():
    raw_config = {
        "type": "sensitivity",
        "algorithm": "fast",
        "harmonics": 1,
        "sample_size": 10,
    }
    yield ert3.config.load_experiment_config(raw_config)


@pytest.fixture()
def gaussian_parameters_config():
    raw_config = [
        {
            "name": "coefficients",
            "type": "stochastic",
            "distribution": {"type": "gaussian", "input": {"mean": 0, "std": 1}},
            "variables": ["a", "b", "c"],
        },
    ]
    yield ert3.config.load_parameters_config(raw_config)


@pytest.fixture()
def uniform_parameters_config():
    raw_config = [
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
    yield ert3.config.load_parameters_config(raw_config)


@pytest.fixture()
def x_uncertainty_parameters_config():
    raw_config = [
        {
            "name": "coefficients",
            "type": "stochastic",
            "distribution": {"type": "gaussian", "input": {"mean": 0, "std": 1}},
            "variables": ["a", "b", "c"],
        },
        {
            "name": "x_normals",
            "type": "stochastic",
            "distribution": {"type": "gaussian", "input": {"mean": 0, "std": 1}},
            "size": 10,
        },
    ]
    yield ert3.config.load_parameters_config(raw_config)


@pytest.mark.requires_ert_storage
def test_run_once_polynomial_evaluation(
    workspace_integration,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    ert3.engine.run(
        ensemble,
        stages_config,
        evaluation_experiment_config,
        gaussian_parameters_config,
        workspace,
        "evaluation",
    )
    with pytest.raises(ValueError, match="Experiment evaluation have been carried out"):
        ert3.engine.run(
            ensemble,
            stages_config,
            evaluation_experiment_config,
            gaussian_parameters_config,
            workspace,
            "evaluation",
        )


@pytest.mark.requires_ert_storage
def test_export_not_run(workspace, ensemble, stages_config):
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "evaluation").ensure(dir=True)
    with pytest.raises(ValueError, match="Cannot export experiment"):
        ert3.engine.export(
            pathlib.Path(), "evaluation", ensemble, stages_config, ensemble.size
        )


def _load_export_data(workspace, experiment_name):
    export_file = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / experiment_name / "data.json"
    )
    with open(export_file) as f:
        return json.load(f)


@pytest.mark.requires_ert_storage
def test_export_polynomial_evaluation(
    workspace_integration,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "evaluation").ensure(dir=True)
    print(workspace)
    ert3.engine.run(
        ensemble,
        stages_config,
        evaluation_experiment_config,
        gaussian_parameters_config,
        workspace,
        "evaluation",
    )
    ert3.engine.export(workspace, "evaluation", ensemble, stages_config, ensemble.size)

    export_data = _load_export_data(workspace, "evaluation")
    assert_export(export_data, ensemble, stages_config, gaussian_parameters_config)


@pytest.mark.requires_ert_storage
def test_export_uniform_polynomial_evaluation(
    workspace_integration,
    uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_config,
):
    workspace = workspace_integration
    uni_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "uniform_evaluation"
    uni_dir.ensure(dir=True)
    ert3.engine.run(
        uniform_ensemble,
        stages_config,
        evaluation_experiment_config,
        uniform_parameters_config,
        workspace,
        "uniform_evaluation",
    )
    ert3.engine.export(
        workspace,
        "uniform_evaluation",
        uniform_ensemble,
        stages_config,
        uniform_ensemble.size,
    )

    export_data = _load_export_data(workspace, "uniform_evaluation")
    assert_export(
        export_data, uniform_ensemble, stages_config, uniform_parameters_config
    )


@pytest.mark.requires_ert_storage
def test_export_x_uncertainties_polynomial_evaluation(
    workspace_integration,
    x_uncertainty_ensemble,
    x_uncertainty_stages_config,
    evaluation_experiment_config,
    x_uncertainty_parameters_config,
):
    workspace = workspace_integration
    uni_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "x_uncertainty"
    uni_dir.ensure(dir=True)
    ert3.engine.run(
        x_uncertainty_ensemble,
        x_uncertainty_stages_config,
        evaluation_experiment_config,
        x_uncertainty_parameters_config,
        workspace,
        "x_uncertainty",
    )
    ert3.engine.export(
        workspace,
        "x_uncertainty",
        x_uncertainty_ensemble,
        x_uncertainty_stages_config,
        x_uncertainty_ensemble.size,
    )

    export_data = _load_export_data(workspace, "x_uncertainty")
    assert_export(
        export_data,
        x_uncertainty_ensemble,
        x_uncertainty_stages_config,
        x_uncertainty_parameters_config,
    )


def test_gaussian_distribution(
    big_ensemble,
    stages_config,
    gaussian_parameters_config,
):
    orig_method = ert3.stats.Gaussian.sample
    returned_samples = []

    def wrapper(self):
        retval = orig_method(self)
        returned_samples.append(retval)
        return retval

    with patch(
        "ert3.stats.Gaussian.sample", side_effect=wrapper, autospec=True
    ) as sample_calls:
        coefficients = ert3.engine.sample_record(
            gaussian_parameters_config, "coefficients", 1000
        )

        assert 1000 == coefficients.ensemble_size
        assert 1000 == len(sample_calls.call_args_list)
        assert 1000 == len(returned_samples)

    assert_distribution(
        big_ensemble,
        stages_config,
        gaussian_parameters_config,
        "gaussian",
        coefficients,
        returned_samples,
    )


def test_uniform_distribution(
    presampled_big_ensemble,
    stages_config,
    uniform_parameters_config,
):
    orig_method = ert3.stats.Uniform.sample
    returned_samples = []

    def wrapper(self):
        retval = orig_method(self)
        returned_samples.append(retval)
        return retval

    with patch(
        "ert3.stats.Uniform.sample", side_effect=wrapper, autospec=True
    ) as sample_calls:
        coefficients = ert3.engine.sample_record(
            uniform_parameters_config,
            "uniform_coefficients",
            1000,
        )

        assert 1000 == coefficients.ensemble_size
        assert 1000 == len(sample_calls.call_args_list)
        assert 1000 == len(returned_samples)

    assert_distribution(
        presampled_big_ensemble,
        stages_config,
        uniform_parameters_config,
        "uniform",
        coefficients,
        returned_samples,
    )


@pytest.mark.requires_ert_storage
def test_run_presampled(
    workspace_integration,
    presampled_ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    presampled_dir = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / "presampled_evaluation"
    )
    presampled_dir.ensure(dir=True)
    coeff0 = ert3.engine.sample_record(gaussian_parameters_config, "coefficients", 10)
    future = ert.storage.transmit_record_collection(
        record_coll=coeff0, record_name="coefficients0", workspace=workspace
    )
    get_event_loop().run_until_complete(future)

    assert 10 == coeff0.ensemble_size
    for real_coeff in coeff0.records:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
        for idx in real_coeff.index:
            assert isinstance(real_coeff.data[idx], float)

    ert3.engine.run(
        presampled_ensemble,
        stages_config,
        evaluation_experiment_config,
        gaussian_parameters_config,
        workspace,
        "presampled_evaluation",
    )
    ert3.engine.export(
        workspace,
        "presampled_evaluation",
        presampled_ensemble,
        stages_config,
        presampled_ensemble.size,
    )

    export_data = _load_export_data(workspace, "presampled_evaluation")
    assert coeff0.ensemble_size == len(export_data)
    for coeff, real in zip(coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.requires_ert_storage
def test_run_uniform_presampled(
    workspace_integration,
    presampled_uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_config,
):
    workspace = workspace_integration
    presampled_dir = (
        workspace / ert3.workspace.EXPERIMENTS_BASE / "presampled_uniform_evaluation"
    )
    presampled_dir.ensure(dir=True)
    uniform_coeff0 = ert3.engine.sample_record(
        uniform_parameters_config,
        "uniform_coefficients",
        10,
    )

    future = ert.storage.transmit_record_collection(
        record_coll=uniform_coeff0,
        record_name="uniform_coefficients0",
        workspace=workspace,
    )
    get_event_loop().run_until_complete(future)
    assert 10 == uniform_coeff0.ensemble_size
    for real_coeff in uniform_coeff0.records:
        assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
        for idx in real_coeff.index:
            assert isinstance(real_coeff.data[idx], float)

    ert3.engine.run(
        presampled_uniform_ensemble,
        stages_config,
        evaluation_experiment_config,
        uniform_parameters_config,
        workspace,
        "presampled_uniform_evaluation",
    )
    ert3.engine.export(
        workspace,
        "presampled_uniform_evaluation",
        presampled_uniform_ensemble,
        stages_config,
        presampled_uniform_ensemble.size,
    )

    export_data = _load_export_data(workspace, "presampled_uniform_evaluation")
    assert uniform_coeff0.ensemble_size == len(export_data)
    for coeff, real in zip(uniform_coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


def test_sample_unknown_parameter_group(uniform_parameters_config):

    with pytest.raises(ValueError, match="No parameter group found named: coeffs"):
        ert3.engine.sample_record(uniform_parameters_config, "coeffs", 100)


@pytest.mark.requires_ert_storage
def test_record_load_and_run(
    workspace_integration,
    doe_ensemble,
    stages_config,
    evaluation_experiment_config,
    designed_coeffs_record_file_integration,
    gaussian_parameters_config,
):
    workspace = workspace_integration

    ert3.engine.load_record(
        workspace,
        "designed_coefficients",
        designed_coeffs_record_file_integration,
        "application/json",
    )

    ert3.engine.run(
        doe_ensemble,
        stages_config,
        evaluation_experiment_config,
        gaussian_parameters_config,
        workspace,
        "doe",
    )
    ert3.engine.export(workspace, "doe", doe_ensemble, stages_config, doe_ensemble.size)

    designed_collection = ert.data.load_collection_from_file(
        designed_coeffs_record_file_integration, "application/json"
    )
    export_data = _load_export_data(workspace, "doe")
    assert designed_collection.ensemble_size == len(export_data)
    for coeff, real in zip(designed_collection.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.requires_ert_storage
def test_record_load_twice(workspace, designed_coeffs_record_file):
    ert3.engine.load_record(
        workspace,
        "designed_coefficients",
        designed_coeffs_record_file,
        "application/json",
    )
    with pytest.raises(ert.exceptions.ElementExistsError):
        ert3.engine.load_record(
            workspace,
            "designed_coefficients",
            designed_coeffs_record_file,
            "application/json",
        )


@pytest.mark.requires_ert_storage
def test_sensitivity_oat_run_and_export(
    workspace_integration,
    sensitivity_ensemble,
    stages_config,
    sensitivity_oat_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "sensitivity").ensure(dir=True)
    ert3.engine.run_sensitivity_analysis(
        sensitivity_ensemble,
        stages_config,
        sensitivity_oat_experiment_config,
        gaussian_parameters_config,
        workspace,
        "sensitivity",
    )
    ensemble_size = ert3.engine.get_ensemble_size(
        ensemble_config=sensitivity_ensemble,
        stages_config=stages_config,
        experiment_config=sensitivity_oat_experiment_config,
        parameters_config=gaussian_parameters_config,
    )
    ert3.engine.export(
        workspace, "sensitivity", sensitivity_ensemble, stages_config, ensemble_size
    )
    export_data = _load_export_data(workspace, "sensitivity")
    assert_sensitivity_export(
        export_data,
        sensitivity_ensemble,
        stages_config,
        gaussian_parameters_config,
        "one-at-a-time",
    )


@pytest.mark.requires_ert_storage
def test_sensitivity_fast_run_and_export(
    workspace_integration,
    sensitivity_ensemble,
    stages_config,
    sensitivity_fast_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    (workspace / ert3.workspace.EXPERIMENTS_BASE / "sensitivity").ensure(dir=True)
    ert3.engine.run_sensitivity_analysis(
        sensitivity_ensemble,
        stages_config,
        sensitivity_fast_experiment_config,
        gaussian_parameters_config,
        workspace,
        "sensitivity",
    )
    ensemble_size = ert3.engine.get_ensemble_size(
        ensemble_config=sensitivity_ensemble,
        stages_config=stages_config,
        experiment_config=sensitivity_fast_experiment_config,
        parameters_config=gaussian_parameters_config,
    )
    ert3.engine.export(
        workspace, "sensitivity", sensitivity_ensemble, stages_config, ensemble_size
    )

    export_data = _load_export_data(workspace, "sensitivity")
    assert_sensitivity_export(
        export_data,
        sensitivity_ensemble,
        stages_config,
        gaussian_parameters_config,
        "fast",
    )


@pytest.mark.requires_ert_storage
def test_partial_sensitivity_run_and_export(
    workspace_integration,
    partial_sensitivity_ensemble,
    double_stages_config,
    sensitivity_oat_experiment_config,
    oat_compatible_record_file,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    experiment_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "partial_sensitivity"
    experiment_dir.ensure(dir=True)
    ert3.engine.load_record(
        workspace, "other_coefficients", oat_compatible_record_file, "application/json"
    )
    ert3.engine.run_sensitivity_analysis(
        partial_sensitivity_ensemble,
        double_stages_config,
        sensitivity_oat_experiment_config,
        gaussian_parameters_config,
        workspace,
        "partial_sensitivity",
    )
    ensemble_size = ert3.engine.get_ensemble_size(
        ensemble_config=partial_sensitivity_ensemble,
        stages_config=double_stages_config,
        experiment_config=sensitivity_oat_experiment_config,
        parameters_config=gaussian_parameters_config,
    )

    ert3.engine.export(
        workspace,
        "partial_sensitivity",
        partial_sensitivity_ensemble,
        double_stages_config,
        ensemble_size,
    )

    export_data = _load_export_data(workspace, "partial_sensitivity")
    assert_sensitivity_export(
        export_data,
        partial_sensitivity_ensemble,
        double_stages_config,
        gaussian_parameters_config,
        "one-at-a-time",
    )


@pytest.mark.requires_ert_storage
def test_incompatible_partial_sensitivity_run(
    workspace_integration,
    partial_sensitivity_ensemble,
    double_stages_config,
    sensitivity_oat_experiment_config,
    oat_incompatible_record_file,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    experiment_dir = workspace / ert3.workspace.EXPERIMENTS_BASE / "partial_sensitivity"
    experiment_dir.ensure(dir=True)
    ert3.engine.load_record(
        workspace,
        "other_coefficients",
        oat_incompatible_record_file,
        "application/json",
    )

    err_msg = "Ensemble size 6 does not match stored record ensemble size 10"
    with pytest.raises(ert.exceptions.ErtError, match=err_msg):
        ert3.engine.run_sensitivity_analysis(
            partial_sensitivity_ensemble,
            double_stages_config,
            sensitivity_oat_experiment_config,
            gaussian_parameters_config,
            workspace,
            "partial_sensitivity",
        )
