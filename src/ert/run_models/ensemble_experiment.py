from __future__ import annotations

import logging
from typing import ClassVar, cast
from uuid import UUID

from pydantic import PrivateAttr

from ert.config import (
    ParameterConfig,
    PostExperimentFixtures,
    PreExperimentFixtures,
    ResponseConfig,
)
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.initial_ensemble_run_model import (
    InitialEnsembleRunModel,
    InitialEnsembleRunModelConfig,
)
from ert.storage import Ensemble
from ert.trace import tracer

logger = logging.getLogger(__name__)


class EnsembleExperimentConfig(InitialEnsembleRunModelConfig):
    target_ensemble: str
    supports_rerunning_failed_realizations: ClassVar[bool] = True


class EnsembleExperiment(InitialEnsembleRunModel, EnsembleExperimentConfig):
    """
    This workflow will create a new experiment and a new ensemble from
    the user configuration.<br>It will never overwrite existing ensembles, and
    will always sample parameters.<br>
    """

    _ensemble_id: UUID | None = PrivateAttr(None)

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

        experiment_storage = self._storage.create_experiment(
            parameters=cast(list[ParameterConfig], self.parameter_configuration),
            observations=self.observation_dataframes(),
            responses=cast(list[ResponseConfig], self.response_configuration),
            name=self.experiment_name,
            templates=self.ert_templates,
        )

        ensemble_storage = self._storage.create_ensemble(
            experiment_storage,
            ensemble_size=self.ensemble_size,
            name=self.target_ensemble,
        )

        self._ensemble_id = ensemble_storage.id
        self.set_env_key("_ERT_EXPERIMENT_ID", str(ensemble_storage.experiment.id))
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble_storage.id))

        self._sample_and_evaluate_ensemble(evaluator_server_config, ensemble_storage)

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
