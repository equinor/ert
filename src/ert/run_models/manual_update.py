from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble

from .base_run_model import ErtRunError, UpdateRunModel

logger = logging.getLogger(__name__)


class ManualUpdate(UpdateRunModel):
    ensemble_id: str

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
        self.set_env_key("_ERT_EXPERIMENT_ID", str(self._prior.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(self._prior.id))

        ensemble_format = self.target_ensemble
        self.update(self._prior, ensemble_format % (self._prior.iteration + 1))

    @classmethod
    def name(cls) -> str:
        return "Manual update"

    @classmethod
    def description(cls) -> str:
        return "Load parameters and responses from existing → update"

    def check_if_runpath_exists(self) -> bool:
        # Will not run a forward model, so does not create files on runpath
        return False
