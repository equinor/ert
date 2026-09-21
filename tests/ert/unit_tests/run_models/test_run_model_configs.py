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

UPDATE_CONFIGS = [
    UpdateRunModelConfig,
    EnsembleSmootherConfig,
    EnsembleInformationFilterConfig,
    MultipleDataAssimilationConfig,
    ManualUpdateConfig,
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
