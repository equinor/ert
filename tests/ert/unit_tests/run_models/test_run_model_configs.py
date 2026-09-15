from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from ert.config import ESSettings, GenKwConfig, ModelConfig, ObservationSettings
from ert.config.queue_config import QueueConfig
from ert.run_models.run_model_configs import (
    EnsembleExperimentConfig,
    EnsembleInformationFilterConfig,
    EnsembleSmootherConfig,
    ManualUpdateConfig,
    MultipleDataAssimilationConfig,
    RunModelConfig,
    SingleTestRunConfig,
    UpdateRunModelConfig,
)

UPDATE_CONFIGS = [
    UpdateRunModelConfig,
    EnsembleSmootherConfig,
    EnsembleInformationFilterConfig,
    MultipleDataAssimilationConfig,
    ManualUpdateConfig,
]


@pytest.fixture
def config_kwargs():
    fields = {
        "storage_path": "storage",
        "runpath_file": Path(".ert_runpath_list"),
        "user_config_file": Path("config.ert"),
        "env_vars": {},
        "env_pr_fm_step": {},
        "runpath_config": ModelConfig(num_realizations=3),
        "queue_config": QueueConfig(),
        "forward_model_steps": [],
        "substitutions": {},
        "hooked_workflows": defaultdict(list),
        "active_realizations": [True, False, True],
        "log_path": Path("logs"),
        "random_seed": 1234,
        "target_ensemble": "ensemble_%d",
        "analysis_settings": ESSettings(),
        "update_settings": ObservationSettings(),
        "experiment_name": "experiment",
        "parameter_configuration": [
            GenKwConfig(
                name="PARAM", distribution={"name": "normal", "mean": 0, "std": 1}
            )
        ],
        "response_configuration": [],
        "ert_templates": [],
        "prior_ensemble_id": None,
        "ensemble_id": "00000000-0000-0000-0000-000000000001",
    }

    def kwargs_for(config_type: type[RunModelConfig]):
        return {
            key: value
            for key, value in fields.items()
            if key in config_type.model_fields
        }

    return kwargs_for


@pytest.mark.parametrize("config_type", UPDATE_CONFIGS)
@pytest.mark.parametrize("active_realizations", [[], [False, False], [False, True]])
def test_that_update_configs_reject_fewer_than_two_active_realizations(
    config_kwargs, config_type, active_realizations
):
    kwargs = config_kwargs(config_type)
    kwargs["active_realizations"] = active_realizations
    with pytest.raises(
        ValidationError, match="Number of active realizations must be at least 2"
    ):
        config_type(**kwargs)


@pytest.mark.parametrize("config_type", UPDATE_CONFIGS)
def test_that_update_configs_accept_two_active_realizations(config_kwargs, config_type):
    config = config_type(**config_kwargs(config_type))
    assert config.active_realizations == [True, False, True]


@pytest.mark.parametrize("config_type", [EnsembleExperimentConfig, SingleTestRunConfig])
def test_that_non_update_configs_accept_one_active_realization(
    config_kwargs, config_type
):
    kwargs = config_kwargs(config_type)
    kwargs["active_realizations"] = [True]
    assert config_type(**kwargs).active_realizations == [True]


@pytest.mark.parametrize("active_realizations", [[], [False], [False, True]])
def test_that_single_test_config_rejects_missing_or_inactive_first_realization(
    config_kwargs, active_realizations
):
    kwargs = config_kwargs(SingleTestRunConfig)
    kwargs["active_realizations"] = active_realizations
    with pytest.raises(ValidationError, match="first realization is inactive"):
        SingleTestRunConfig(**kwargs)


def test_that_single_test_config_defaults_to_realization_zero(config_kwargs):
    kwargs = config_kwargs(SingleTestRunConfig)
    del kwargs["active_realizations"]
    assert SingleTestRunConfig(**kwargs).active_realizations == [True]


@pytest.mark.parametrize(
    "config_type",
    [
        EnsembleSmootherConfig,
        EnsembleInformationFilterConfig,
        MultipleDataAssimilationConfig,
    ],
)
@pytest.mark.parametrize("has_parameters", [False, True], ids=["empty", "all-disabled"])
def test_that_initial_update_configs_require_updatable_parameters(
    config_kwargs, config_type, has_parameters
):
    kwargs = config_kwargs(config_type)
    for parameter in kwargs["parameter_configuration"]:
        parameter.update_strategy = None
    if not has_parameters:
        kwargs["parameter_configuration"] = []

    with pytest.raises(ValidationError, match="No parameters to update"):
        config_type(**kwargs)


@pytest.mark.parametrize(
    "config_type",
    [
        EnsembleSmootherConfig,
        EnsembleInformationFilterConfig,
        MultipleDataAssimilationConfig,
    ],
)
def test_that_initial_update_configs_accept_mix_of_fixed_and_updatable_parameters(
    config_kwargs, config_type
):
    kwargs = config_kwargs(config_type)
    fixed = GenKwConfig(
        name="FIXED",
        distribution={"name": "normal", "mean": 0, "std": 1},
        update_strategy=None,
    )
    kwargs["parameter_configuration"].append(fixed)
    assert len(config_type(**kwargs).parameter_configuration) == 2


@pytest.mark.parametrize("has_parameters", [False, True], ids=["empty", "all-disabled"])
def test_that_restart_configs_allow_current_parameters_without_updates(
    config_kwargs, has_parameters
):
    kwargs = config_kwargs(MultipleDataAssimilationConfig)
    kwargs["prior_ensemble_id"] = "00000000-0000-0000-0000-000000000001"
    for parameter in kwargs["parameter_configuration"]:
        parameter.update_strategy = None
    if not has_parameters:
        kwargs["parameter_configuration"] = []
    config = MultipleDataAssimilationConfig(**kwargs)
    assert config.parameter_configuration == kwargs["parameter_configuration"]


@pytest.mark.parametrize("config_type", [EnsembleExperimentConfig, SingleTestRunConfig])
def test_that_non_update_configs_allow_no_parameters(config_kwargs, config_type):
    kwargs = config_kwargs(config_type)
    kwargs["parameter_configuration"] = []
    assert config_type(**kwargs).parameter_configuration == []
