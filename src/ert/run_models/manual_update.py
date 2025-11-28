from __future__ import annotations

import functools
import logging
from typing import Any
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.update_run_model import UpdateRunModel, UpdateRunModelConfig
from ert.storage import Ensemble

from ..analysis import smoother_update
from .run_model import ErtRunError

logger = logging.getLogger(__name__)


class ManualUpdateConfig(UpdateRunModelConfig):
    ensemble_id: str
    ert_templates: list[tuple[str, str]]


class ManualUpdate(UpdateRunModel, ManualUpdateConfig):
    _prior: Ensemble = PrivateAttr()

    def model_post_init(self, ctx: Any) -> None:
        super().model_post_init(ctx)

        try:
            self._prior = self._storage.get_ensemble(UUID(self.ensemble_id))
        except (KeyError, ValueError) as err:
            raise ErtRunError(
                f"Prior ensemble with ID: {UUID(self.ensemble_id)} does not exist"
            ) from err

    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        prior_experiment = self._prior.experiment

        self.set_env_key("_ERT_EXPERIMENT_ID", str(prior_experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(self._prior.id))

        target_experiment = self._storage.create_experiment(
            parameters=list(prior_experiment.parameter_configuration.values()),
            responses=list(prior_experiment.response_configuration.values()),
            observations=prior_experiment.observations,
            simulation_arguments=prior_experiment.metadata,
            name=f"Manual update of {self._prior.name}",
            templates=self.ert_templates,
        )
        self.update(
            self._prior,
            self.target_ensemble % (self._prior.iteration + 1),
            target_experiment=target_experiment,
        )

    def update_ensemble_parameters(
        self, prior: Ensemble, posterior: Ensemble, weight: float
    ) -> None:
        smoother_update(
            prior,
            posterior,
            update_settings=self.update_settings,
            es_settings=self.analysis_settings,
            parameters=prior.experiment.update_parameters,
            observations=prior.experiment.observation_keys,
            global_scaling=weight,
            rng=self._rng,
            progress_callback=functools.partial(
                self.send_smoother_event,
                prior.iteration,
                prior.id,
            ),
            active_realizations=self.active_realizations,
        )

    @classmethod
    def name(cls) -> str:
        return "Manual update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → update"

    def check_if_runpath_exists(self) -> bool:
        # Will not run a forward model, so does not create files on runpath
        return False
