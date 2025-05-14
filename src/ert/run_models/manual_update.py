from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import Ensemble, open_storage

from .base_run_model import ErtRunError, UpdateRunModel

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class ManualUpdate(UpdateRunModel):
    ensemble_id: str
    target_ensemble: str
    support_restart: bool = False

    _prior: Ensemble = PrivateAttr()

    def model_post_init(self, ctx) -> None:
        try:
            with open_storage(self.storage_path, mode="r") as storage:
                prior = storage.get_ensemble(UUID(self.ensemble_id))
        except (KeyError, ValueError) as err:
            raise ErtRunError(
                f"Prior ensemble with ID: {UUID(self.ensemble_id)} does not exists"
            ) from err

        self._prior = prior
        self.support_restart = False
        super().model_post_init(ctx)

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig, restart: bool = False
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
