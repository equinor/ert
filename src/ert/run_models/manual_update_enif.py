from __future__ import annotations

import functools
import logging
from typing import Literal

from ert.analysis import enif_update
from ert.storage import Ensemble

from .manual_update import ManualUpdate, ManualUpdateConfig

logger = logging.getLogger(__name__)


class ManualUpdateEnIFConfig(ManualUpdateConfig):
    type: Literal["manual_update_enif"] = "manual_update_enif"


class ManualUpdateEnIF(ManualUpdateEnIFConfig, ManualUpdate):
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
