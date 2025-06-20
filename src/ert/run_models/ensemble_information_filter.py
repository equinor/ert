from __future__ import annotations

import functools
import logging

from ert.run_models.ensemble_smoother import EnsembleSmoother
from ert.storage import Ensemble

from ..analysis import enif_update

logger = logging.getLogger(__name__)


class EnsembleInformationFilter(EnsembleSmoother):
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

    @classmethod
    def name(cls) -> str:
        return "Ensemble Information Filter (Experimental)"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate → EnIF update → evaluate"
