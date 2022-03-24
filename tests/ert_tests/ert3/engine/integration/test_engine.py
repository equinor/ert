import json
import pathlib
from contextlib import contextmanager
import resource

import pytest

from ert_shared.async_utils import get_event_loop
from integration_utils import (
    assert_distribution,
    assert_export,
    assert_sensitivity_export,
)

import ert
import ert3
from unittest.mock import patch

_EXPERIMENTS_BASE = ert3.workspace._workspace._EXPERIMENTS_BASE
_RESOURCES_BASE = ert3.workspace._workspace._RESOURCES_BASE


@pytest.fixture()
def sensitivity_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict.pop("size")
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def uniform_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "stochastic.uniform_coefficients"
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def x_uncertainty_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["forward_model"]["stage"] = "evaluate_x_uncertainty_polynomial"
    base_ensemble_dict["input"].append(
        {"record": "x_uncertainties", "source": "stochastic.x_normals"}
    )
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def presampled_uniform_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "storage.uniform_coefficients0"
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def presampled_loguniform_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "storage.loguniform_coefficients0"
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def presampled_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "storage.coefficients0"
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def doe_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "storage.designed_coefficients"
    base_ensemble_dict["size"] = 10
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def big_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "storage.coefficients0"
    base_ensemble_dict["size"] = 1000
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def presampled_big_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict["input"][0]["source"] = "storage.uniform_coefficients0"
    base_ensemble_dict["size"] = 1000
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


@pytest.fixture()
def partial_sensitivity_ensemble(base_ensemble_dict, plugin_registry):
    base_ensemble_dict.pop("size")
    base_ensemble_dict["input"].append(
        {"record": "other_coefficients", "source": "storage.other_coefficients"}
    )
    base_ensemble_dict["output"].append({"record": "other_polynomial_output"})
    yield ert3.config.load_ensemble_config(
        base_ensemble_dict, plugin_registry=plugin_registry
    )


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
def loguniform_parameters_config():
    raw_config = [
        {
            "name": "loguniform_coefficients",
            "type": "stochastic",
            "distribution": {
                "type": "loguniform",
                "input": {"lower_bound": 0.01, "upper_bound": 1},
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


@contextmanager
def assert_clean_workspace(workspace, allowed_files=None):
    def _get_files(workspace):
        files = set(pathlib.Path(workspace._path).rglob("*"))
        dot_ert_files = set(pathlib.Path(workspace._path / ".ert").rglob("*"))
        return files - dot_ert_files

    files = _get_files(workspace)
    yield
    if allowed_files is None:
        assert files == _get_files(workspace)
    else:
        new_files = _get_files(workspace) - files
        assert set(item.name for item in new_files) == set(allowed_files)


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_run_once_polynomial_evaluation(
    workspace_integration,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    experiment_run_config = ert3.config.ExperimentRunConfig(
        evaluation_experiment_config,
        stages_config,
        ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace):
        ert3.engine.run(experiment_run_config, workspace, "evaluation")
    with assert_clean_workspace(workspace):
        with pytest.raises(
            ert.exceptions.ExperimentError,
            match="Experiment 'evaluation' has been carried out already.",
        ):
            ert3.engine.run(experiment_run_config, workspace, "evaluation")


@pytest.mark.requires_ert_storage
def test_export_not_run(workspace, ensemble, stages_config):
    (workspace._path / _EXPERIMENTS_BASE / "evaluation").mkdir(parents=True)
    experiment_run_config = ert3.config.ExperimentRunConfig(
        ert3.config.ExperimentConfig(type="evaluation"),
        stages_config,
        ensemble,
        ert3.config.ParametersConfig.parse_obj([]),
    )
    with assert_clean_workspace(workspace):
        with pytest.raises(ValueError, match="Cannot export experiment"):
            ert3.engine.export(
                workspace,
                "evaluation",
                experiment_run_config,
            )


def _load_export_data(workspace, experiment_name):
    export_file = workspace._path / _EXPERIMENTS_BASE / experiment_name / "data.json"
    with open(export_file) as f:
        return json.load(f)


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_export_polynomial_evaluation(
    workspace_integration,
    ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    (workspace._path / _EXPERIMENTS_BASE / "evaluation").mkdir(parents=True)
    experiment_run_config = ert3.config.ExperimentRunConfig(
        evaluation_experiment_config,
        stages_config,
        ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace):
        ert3.engine.run(experiment_run_config, workspace, "evaluation")

    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "evaluation", experiment_run_config)

    export_data = _load_export_data(workspace, "evaluation")
    assert_export(export_data, ensemble, stages_config, gaussian_parameters_config)


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_export_uniform_polynomial_evaluation(
    workspace_integration,
    uniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    uniform_parameters_config,
):
    workspace = workspace_integration
    experiment_run_config = ert3.config.ExperimentRunConfig(
        evaluation_experiment_config,
        stages_config,
        uniform_ensemble,
        uniform_parameters_config,
    )
    uni_dir = workspace._path / _EXPERIMENTS_BASE / "uniform_evaluation"
    uni_dir.mkdir(parents=True)
    with assert_clean_workspace(workspace):
        ert3.engine.run(experiment_run_config, workspace, "uniform_evaluation")
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "uniform_evaluation", experiment_run_config)

    export_data = _load_export_data(workspace, "uniform_evaluation")
    assert_export(
        export_data, uniform_ensemble, stages_config, uniform_parameters_config
    )


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_export_x_uncertainties_polynomial_evaluation(
    workspace_integration,
    x_uncertainty_ensemble,
    x_uncertainty_stages_config,
    evaluation_experiment_config,
    x_uncertainty_parameters_config,
):
    workspace = workspace_integration
    experiment_run_config = ert3.config.ExperimentRunConfig(
        evaluation_experiment_config,
        x_uncertainty_stages_config,
        x_uncertainty_ensemble,
        x_uncertainty_parameters_config,
    )
    uni_dir = workspace._path / _EXPERIMENTS_BASE / "x_uncertainty"
    uni_dir.mkdir(parents=True)
    with assert_clean_workspace(workspace):
        ert3.engine.run(experiment_run_config, workspace, "x_uncertainty")
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "x_uncertainty", experiment_run_config)

    export_data = _load_export_data(workspace, "x_uncertainty")
    assert_export(
        export_data,
        x_uncertainty_ensemble,
        x_uncertainty_stages_config,
        x_uncertainty_parameters_config,
    )


@pytest.mark.usefixtures("setup_tmpdir")
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

        assert 1000 == len(coefficients)
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


@pytest.mark.usefixtures("setup_tmpdir")
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

        assert 1000 == len(coefficients)
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


@pytest.mark.usefixtures("setup_tmpdir")
def test_loguniform_distribution(
    presampled_big_ensemble,
    stages_config,
    loguniform_parameters_config,
):
    orig_method = ert3.stats.LogUniform.sample
    returned_samples = []

    def wrapper(self):
        retval = orig_method(self)
        returned_samples.append(retval)
        return retval

    with patch(
        "ert3.stats.LogUniform.sample", side_effect=wrapper, autospec=True
    ) as sample_calls:
        coefficients = ert3.engine.sample_record(
            loguniform_parameters_config,
            "loguniform_coefficients",
            1000,
        )

        assert 1000 == len(coefficients)
        assert 1000 == len(sample_calls.call_args_list)
        assert 1000 == len(returned_samples)

    assert_distribution(
        presampled_big_ensemble,
        stages_config,
        loguniform_parameters_config,
        "loguniform",
        coefficients,
        returned_samples,
    )


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_run_presampled(
    workspace_integration,
    presampled_ensemble,
    stages_config,
    evaluation_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    presampled_dir = workspace._path / _EXPERIMENTS_BASE / "presampled_evaluation"
    presampled_dir.mkdir(parents=True)
    with assert_clean_workspace(workspace):
        coeff0 = ert3.engine.sample_record(
            gaussian_parameters_config, "coefficients", 10
        )
        future = ert.storage.transmit_record_collection(
            record_coll=coeff0,
            record_name="coefficients0",
            workspace_name=workspace.name,
        )
        get_event_loop().run_until_complete(future)

        assert 10 == len(coeff0)
        for real_coeff in coeff0.records:
            assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
            for idx in real_coeff.index:
                assert isinstance(real_coeff.data[idx], float)

        experiment_run_config = ert3.config.ExperimentRunConfig(
            evaluation_experiment_config,
            stages_config,
            presampled_ensemble,
            gaussian_parameters_config,
        )
        ert3.engine.run(experiment_run_config, workspace, "presampled_evaluation")
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "presampled_evaluation", experiment_run_config)

    export_data = _load_export_data(workspace, "presampled_evaluation")
    assert len(coeff0) == len(export_data)
    for coeff, real in zip(coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.integration_test
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
        workspace._path / _EXPERIMENTS_BASE / "presampled_uniform_evaluation"
    )
    presampled_dir.mkdir(parents=True)
    with assert_clean_workspace(workspace):
        uniform_coeff0 = ert3.engine.sample_record(
            uniform_parameters_config,
            "uniform_coefficients",
            10,
        )

        future = ert.storage.transmit_record_collection(
            record_coll=uniform_coeff0,
            record_name="uniform_coefficients0",
            workspace_name=workspace.name,
        )
        get_event_loop().run_until_complete(future)
        assert 10 == len(uniform_coeff0)
        for real_coeff in uniform_coeff0.records:
            assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
            for idx in real_coeff.index:
                assert isinstance(real_coeff.data[idx], float)

        experiment_run_config = ert3.config.ExperimentRunConfig(
            evaluation_experiment_config,
            stages_config,
            presampled_uniform_ensemble,
            uniform_parameters_config,
        )
        ert3.engine.run(
            experiment_run_config, workspace, "presampled_uniform_evaluation"
        )
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(
            workspace, "presampled_uniform_evaluation", experiment_run_config
        )

    export_data = _load_export_data(workspace, "presampled_uniform_evaluation")
    assert len(uniform_coeff0) == len(export_data)
    for coeff, real in zip(uniform_coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_run_loguniform_presampled(
    workspace_integration,
    presampled_loguniform_ensemble,
    stages_config,
    evaluation_experiment_config,
    loguniform_parameters_config,
):
    workspace = workspace_integration
    presampled_dir = (
        workspace._path / _EXPERIMENTS_BASE / "presampled_loguniform_evaluation"
    )
    presampled_dir.mkdir(parents=True)
    with assert_clean_workspace(workspace):
        loguniform_coeff0 = ert3.engine.sample_record(
            loguniform_parameters_config,
            "loguniform_coefficients",
            10,
        )

        future = ert.storage.transmit_record_collection(
            record_coll=loguniform_coeff0,
            record_name="loguniform_coefficients0",
            workspace_name=workspace.name,
        )
        get_event_loop().run_until_complete(future)
        assert 10 == len(loguniform_coeff0)
        for real_coeff in loguniform_coeff0.records:
            assert sorted(("a", "b", "c")) == sorted(real_coeff.index)
            for idx in real_coeff.index:
                assert isinstance(real_coeff.data[idx], float)

        experiment_run_config = ert3.config.ExperimentRunConfig(
            evaluation_experiment_config,
            stages_config,
            presampled_loguniform_ensemble,
            loguniform_parameters_config,
        )
        ert3.engine.run(
            experiment_run_config, workspace, "presampled_loguniform_evaluation"
        )
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(
            workspace, "presampled_loguniform_evaluation", experiment_run_config
        )

    export_data = _load_export_data(workspace, "presampled_loguniform_evaluation")
    assert len(loguniform_coeff0) == len(export_data)
    for coeff, real in zip(loguniform_coeff0.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


def test_sample_unknown_parameter_group(uniform_parameters_config):

    with pytest.raises(ValueError, match="No parameter group found named: coeffs"):
        ert3.engine.sample_record(uniform_parameters_config, "coeffs", 100)


@pytest.mark.integration_test
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
    with assert_clean_workspace(workspace):
        transformation = ert.data.SerializationTransformation(
            location=designed_coeffs_record_file_integration, mime="application/json"
        )
        get_event_loop().run_until_complete(
            ert3.engine.load_record(
                workspace,
                "designed_coefficients",
                transformation,
            )
        )

        experiment_run_config = ert3.config.ExperimentRunConfig(
            evaluation_experiment_config,
            stages_config,
            doe_ensemble,
            gaussian_parameters_config,
        )
        ert3.engine.run(experiment_run_config, workspace, "doe")
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "doe", experiment_run_config)

    transformation = ert.data.SerializationTransformation(
        location=designed_coeffs_record_file_integration,
        mime="application/json",
    )
    designed_collection = get_event_loop().run_until_complete(
        ert.data.load_collection_from_file(transformation)
    )
    export_data = _load_export_data(workspace, "doe")
    assert len(designed_collection) == len(export_data)
    for coeff, real in zip(designed_collection.records, export_data):
        assert ["coefficients"] == list(real["input"].keys())
        export_coeff = real["input"]["coefficients"]
        assert sorted(coeff.index) == sorted(export_coeff.keys())
        for key in coeff.index:
            assert coeff.data[key] == export_coeff[key]


@pytest.mark.requires_ert_storage
@pytest.mark.asyncio
async def test_record_load_twice(workspace, designed_coeffs_record_file):
    with assert_clean_workspace(workspace):
        transformation = ert.data.SerializationTransformation(
            location=designed_coeffs_record_file,
            mime="application/json",
        )
        await ert3.engine.load_record(
            workspace,
            "designed_coefficients",
            transformation,
        )
        with pytest.raises(ert.exceptions.ElementExistsError):
            await ert3.engine.load_record(
                workspace,
                "designed_coefficients",
                transformation,
            )


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_sensitivity_oat_run_and_export(
    workspace_integration,
    sensitivity_ensemble,
    stages_config,
    sensitivity_oat_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    (workspace._path / _EXPERIMENTS_BASE / "sensitivity").mkdir(parents=True)
    experiment_run_config = ert3.config.ExperimentRunConfig(
        sensitivity_oat_experiment_config,
        stages_config,
        sensitivity_ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace):
        ert3.engine.run_sensitivity_analysis(
            experiment_run_config, workspace, "sensitivity"
        )
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "sensitivity", experiment_run_config)
    export_data = _load_export_data(workspace, "sensitivity")
    assert_sensitivity_export(
        export_data,
        sensitivity_ensemble,
        stages_config,
        gaussian_parameters_config,
        "one-at-a-time",
    )


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_sensitivity_fast_run_and_export(
    workspace_integration,
    sensitivity_ensemble,
    stages_config,
    sensitivity_fast_experiment_config,
    gaussian_parameters_config,
):
    workspace = workspace_integration
    (workspace._path / _EXPERIMENTS_BASE / "sensitivity").mkdir(parents=True)
    experiment_run_config = ert3.config.ExperimentRunConfig(
        sensitivity_fast_experiment_config,
        stages_config,
        sensitivity_ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace, allowed_files={"fast_analysis.json"}):
        ert3.engine.run_sensitivity_analysis(
            experiment_run_config, workspace, "sensitivity"
        )
    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "sensitivity", experiment_run_config)

    export_data = _load_export_data(workspace, "sensitivity")
    assert_sensitivity_export(
        export_data,
        sensitivity_ensemble,
        stages_config,
        gaussian_parameters_config,
        "fast",
    )


@pytest.mark.integration_test
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
    experiment_dir = workspace._path / _EXPERIMENTS_BASE / "partial_sensitivity"
    assert experiment_dir.is_dir()
    experiment_run_config = ert3.config.ExperimentRunConfig(
        sensitivity_oat_experiment_config,
        double_stages_config,
        partial_sensitivity_ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace):
        transformation = ert.data.SerializationTransformation(
            location=oat_compatible_record_file,
            mime="application/json",
        )
        get_event_loop().run_until_complete(
            ert3.engine.load_record(
                workspace,
                "other_coefficients",
                transformation,
            )
        )
        ert3.engine.run_sensitivity_analysis(
            experiment_run_config, workspace, "partial_sensitivity"
        )

    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "partial_sensitivity", experiment_run_config)

    export_data = _load_export_data(workspace, "partial_sensitivity")
    assert_sensitivity_export(
        export_data,
        partial_sensitivity_ensemble,
        double_stages_config,
        gaussian_parameters_config,
        "one-at-a-time",
    )


@pytest.mark.integration_test
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
    experiment_dir = workspace._path / _EXPERIMENTS_BASE / "partial_sensitivity"
    assert experiment_dir.is_dir()
    with assert_clean_workspace(workspace):
        transformation = ert.data.SerializationTransformation(
            location=oat_incompatible_record_file,
            mime="application/json",
        )
        get_event_loop().run_until_complete(
            ert3.engine.load_record(
                workspace,
                "other_coefficients",
                transformation,
            )
        )

    err_msg = (
        "Ensemble size 6 does not match stored record ensemble "
        "for other_coefficients of size 10"
    )
    experiment_run_config = ert3.config.ExperimentRunConfig(
        sensitivity_oat_experiment_config,
        double_stages_config,
        partial_sensitivity_ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace):
        with pytest.raises(ert.exceptions.ErtError, match=err_msg):
            ert3.engine.run_sensitivity_analysis(
                experiment_run_config,
                workspace,
                "partial_sensitivity",
            )


@pytest.mark.integration_test
@pytest.mark.requires_ert_storage
def test_export_resources(
    workspace_integration,
    evaluation_experiment_config,
    plugin_registry,
    gaussian_parameters_config,
):
    """Test that resources will be exported if they are numerical."""
    workspace = workspace_integration
    (workspace._path / _EXPERIMENTS_BASE / "evaluation").mkdir(parents=True)

    resources = workspace._path / _RESOURCES_BASE
    resources.mkdir(parents=True)
    with open(resources / "coefficients.json", "w") as f:
        json.dump([{"a": 0, "b": 0, "c": 0}], f)

    with open(resources / "my_blob", "w") as f:
        f.write("blob blobb")

    stages = ert3.config.load_stages_config(
        [
            {
                "name": "noop",
                "input": [
                    {
                        "name": "coefficients",
                        "transformation": {
                            "location": "coefficients.json",
                            "type": "serialization",
                        },
                    },
                    {
                        "name": "my_blob",
                        "transformation": {
                            "location": "my_blob",
                        },
                    },
                ],
                "output": [],
                "script": [],
                "transportable_commands": [],
            },
        ],
        plugin_registry=plugin_registry,
    )

    ensemble = ert3.config.load_ensemble_config(
        {
            "size": 1,
            "input": [
                {
                    "source": "resources.coefficients.json",
                    "record": "coefficients",
                    "transformation": {"type": "serialization"},
                },
                {
                    "source": "resources.my_blob",
                    "record": "my_blob",
                },
            ],
            "output": [],
            "forward_model": {"driver": "local", "stage": "noop"},
        },
        plugin_registry=plugin_registry,
    )

    experiment_run_config = ert3.config.ExperimentRunConfig(
        evaluation_experiment_config,
        stages,
        ensemble,
        gaussian_parameters_config,
    )
    with assert_clean_workspace(workspace):
        ert3.engine.run(experiment_run_config, workspace, "evaluation")

    with assert_clean_workspace(workspace, allowed_files={"data.json"}):
        ert3.engine.export(workspace, "evaluation", experiment_run_config)

    export_data = _load_export_data(workspace, "evaluation")
    assert "coefficients" in export_data[0]["input"]
    assert "my_blob" not in export_data[0]["input"]
