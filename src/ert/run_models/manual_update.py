from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.experiment_configs import ManualUpdateConfig
from ert.experiment_types import ExperimentType
from ert.run_models.update_run_model import UpdateRunModel
from ert.storage import Ensemble

from .run_model import ErtRunError

logger = logging.getLogger(__name__)


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

        experiment_config = self.model_dump(mode="json") | {
            "parameter_configuration": prior_experiment.experiment_config[
                "parameter_configuration"
            ],
            "response_configuration": prior_experiment.experiment_config[
                "response_configuration"
            ],
            "observations": prior_experiment.experiment_config["observations"],
        }

        target_experiment = self._storage.create_experiment(
            experiment_config=experiment_config,
            name=f"Manual update of {self._prior.name}",
        )
        self.update(
            self._prior,
            self.target_ensemble % (self._prior.iteration + 1),
            target_experiment=target_experiment,
        )

    @classmethod
    def name(cls) -> str:
        return "Manual update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → update"

    @classmethod
    def _experiment_type(cls) -> ExperimentType:
        return ExperimentType.MANUAL_UPDATE

    def check_if_runpath_exists(self) -> bool:
        # Will not run a forward model, so does not create files on runpath
        return False
