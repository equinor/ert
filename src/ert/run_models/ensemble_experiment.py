from __future__ import annotations

import logging
from typing import ClassVar
from uuid import UUID

from pydantic import PrivateAttr

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.plugins import PostExperimentFixtures, PreExperimentFixtures
from ert.run_models.initial_ensemble_run_model import InitialEnsembleRunModel
from ert.storage import Ensemble
from ert.trace import tracer

logger = logging.getLogger(__name__)


class EnsembleExperiment(InitialEnsembleRunModel):
    """
    This workflow will create a new experiment and a new ensemble from
    the user configuration.<br>It will never overwrite existing ensembles, and
    will always sample parameters.<br>
    """

    _ensemble_id: UUID | None = PrivateAttr(None)
    supports_rerunning_failed_realizations: ClassVar[bool] = True
    target_ensemble: str

    @property
    def _ensemble(self) -> Ensemble:
        assert self._ensemble_id is not None
        return self._storage.get_ensemble(self._ensemble_id)

    @tracer.start_as_current_span(f"{__name__}.run_experiment")
    def run_experiment(
        self,
        evaluator_server_config: EvaluatorServerConfig,
        rerun_failed_realizations: bool = False,
    ) -> None:
        self.log_at_startup()
        self._rerun_failed_realizations = rerun_failed_realizations

        self.run_workflows(fixtures=PreExperimentFixtures(random_seed=self.random_seed))

        self._sample_and_evaluate_ensemble(
            evaluator_server_config,
            None,
            self.target_ensemble,
            rerun_failed_realizations,
            self._ensemble if rerun_failed_realizations else None,
        )

        self.run_workflows(
            fixtures=PostExperimentFixtures(
                random_seed=self.random_seed,
                storage=self._storage,
                ensemble=self._ensemble,
            ),
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate all realizations"
