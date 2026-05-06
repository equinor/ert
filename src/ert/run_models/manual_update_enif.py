from __future__ import annotations

import functools
import logging
from typing import Literal

from ert.analysis import enif_update
from ert.run_models.run_model_configs import ManualUpdateConfig
from ert.storage import Ensemble
from ert.storage.local_experiment import ExperimentConfig, ExperimentType

from .manual_update import ManualUpdate

logger = logging.getLogger(__name__)


class ManualUpdateEnIFConfig(ManualUpdateConfig):
    experiment_type: Literal[ExperimentType.MANUAL] = ExperimentType.MANUAL

    def to_experiment_config(
        self, *, prior_experiment_config: ExperimentConfig
    ) -> ExperimentConfig:
        config = super().to_experiment_config(
            prior_experiment_config=prior_experiment_config
        )
        config["experiment_type"] = ExperimentType.MANUAL
        return config


class ManualUpdateEnIF(ManualUpdate):
    @classmethod
    def name(cls) -> str:
        return "Manual EnIF update (Experimental)"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → EnIF update"

    def update_ensemble_parameters(
        self, prior: Ensemble, posterior: Ensemble, weight: float
    ) -> None:
        enif_update(
            prior,
            posterior,
            parameters=prior.experiment.update_parameters,
            observations=prior.experiment.observation_keys,
            random_seed=self.random_seed,
            progress_callback=functools.partial(
                self.send_smoother_event,
                prior.iteration,
                prior.id,
            ),
        )
