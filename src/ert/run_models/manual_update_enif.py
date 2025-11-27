from __future__ import annotations

import functools
import logging

from ert.storage import Ensemble

from ..analysis import enif_update
from .manual_update import ManualUpdate

logger = logging.getLogger(__name__)


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
