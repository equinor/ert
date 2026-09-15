from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from ert.config import (
    ESSettings,
    GenKwConfig,
    LocalizationType,
    ModelConfig,
    ObservationSettings,
)
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

INITIAL_UPDATE_CONFIGS = [
    EnsembleSmootherConfig,
    EnsembleInformationFilterConfig,
    MultipleDataAssimilationConfig,
]
UPDATE_CONFIGS = [
    UpdateRunModelConfig,
    ManualUpdateConfig,
    *INITIAL_UPDATE_CONFIGS,
]
NON_UPDATE_CONFIGS = [EnsembleExperimentConfig, SingleTestRunConfig]

EXISTING_ENSEMBLE_ID = "00000000-0000-0000-0000-000000000001"


def gen_kw_config(
    name: str, update_strategy: LocalizationType | None = LocalizationType.GLOBAL
) -> GenKwConfig:
    return GenKwConfig(
        name=name,
        distribution={"name": "normal", "mean": 0, "std": 1},
        update_strategy=update_strategy,
    )


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
        "parameter_configuration": [gen_kw_config("PARAM")],
        "response_configuration": [],
        "ert_templates": [],
        "prior_ensemble_id": None,
        "ensemble_id": EXISTING_ENSEMBLE_ID,
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


@pytest.mark.parametrize(
    ("config_type", "active_realizations"),
    [(config_type, [True, False, True]) for config_type in UPDATE_CONFIGS]
    + [(config_type, [True]) for config_type in NON_UPDATE_CONFIGS],
)
def test_that_configs_accept_their_minimum_number_of_active_realizations(
    config_kwargs, config_type, active_realizations
):
    kwargs = config_kwargs(config_type)
    kwargs["active_realizations"] = active_realizations
    assert config_type(**kwargs).active_realizations == active_realizations


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


@pytest.mark.parametrize("config_type", INITIAL_UPDATE_CONFIGS)
@pytest.mark.parametrize(
    "num_fixed_parameters", [0, 1], ids=["no-parameters", "only-fixed-parameters"]
)
def test_that_initial_update_configs_require_updatable_parameters(
    config_kwargs, config_type, num_fixed_parameters
):
    kwargs = config_kwargs(config_type)
    kwargs["parameter_configuration"] = [
        gen_kw_config(f"FIXED_{index}", update_strategy=None)
        for index in range(num_fixed_parameters)
    ]

    with pytest.raises(ValidationError, match="No parameters to update"):
        config_type(**kwargs)


@pytest.mark.parametrize("config_type", INITIAL_UPDATE_CONFIGS)
def test_that_initial_update_configs_accept_mix_of_fixed_and_updatable_parameters(
    config_kwargs, config_type
):
    kwargs = config_kwargs(config_type)
    kwargs["parameter_configuration"].append(
        gen_kw_config("FIXED", update_strategy=None)
    )
    assert len(config_type(**kwargs).parameter_configuration) == 2


@pytest.mark.parametrize(
    "num_fixed_parameters", [0, 1], ids=["no-parameters", "only-fixed-parameters"]
)
def test_that_restart_configs_allow_current_parameters_without_updates(
    config_kwargs, num_fixed_parameters
):
    kwargs = config_kwargs(MultipleDataAssimilationConfig)
    kwargs["prior_ensemble_id"] = EXISTING_ENSEMBLE_ID
    kwargs["parameter_configuration"] = [
        gen_kw_config(f"FIXED_{index}", update_strategy=None)
        for index in range(num_fixed_parameters)
    ]
    config = MultipleDataAssimilationConfig(**kwargs)
    assert config.parameter_configuration == kwargs["parameter_configuration"]


@pytest.mark.parametrize("config_type", NON_UPDATE_CONFIGS)
def test_that_non_update_configs_allow_no_parameters(config_kwargs, config_type):
    kwargs = config_kwargs(config_type)
    kwargs["parameter_configuration"] = []
    assert config_type(**kwargs).parameter_configuration == []


@pytest.mark.parametrize("prior_ensemble_id", [None, ""])
def test_that_non_restart_configs_require_an_experiment_name(
    config_kwargs, prior_ensemble_id
):
    kwargs = config_kwargs(MultipleDataAssimilationConfig)
    kwargs.update(prior_ensemble_id=prior_ensemble_id, experiment_name="")
    with pytest.raises(
        ValidationError, match="For non-restart run, experiment name must be set"
    ):
        MultipleDataAssimilationConfig(**kwargs)


def test_that_restart_configs_allow_an_empty_experiment_name(config_kwargs):
    kwargs = config_kwargs(MultipleDataAssimilationConfig)
    kwargs.update(
        prior_ensemble_id=EXISTING_ENSEMBLE_ID,
        experiment_name="",
    )
    assert not MultipleDataAssimilationConfig(**kwargs).experiment_name
