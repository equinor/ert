from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ert.analysis import ErtAnalysisError
from ert.config import HookRuntime
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.libres_facade import LibresFacade
from ert.realization_state import RealizationState
from ert.run_context import RunContext
from ert.run_models.run_arguments import ESRunArguments
from ert.storage import StorageAccessor

from .base_run_model import BaseRunModel, ErtRunError

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.enkf_main import EnKFMain

logger = logging.getLogger(__file__)


# pylint: disable=too-many-arguments
class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: ESRunArguments,
        ert: EnKFMain,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
    ):
        super().__init__(
            simulation_arguments,
            ert,
            LibresFacade(ert),
            storage,
            queue_config,
            experiment_id,
            phase_count=2,
        )
        self.support_restart = False

    def run_experiment(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        self.checkHaveSufficientRealizations(
            self._simulation_arguments.active_realizations.count(True)
        )

        log_msg = "Running ES"
        logger.info(log_msg)
        self.setPhaseName(log_msg, indeterminate=True)

        prior_name = self._simulation_arguments.current_case
        prior = self._storage.create_ensemble(
            self._experiment_id,
            ensemble_size=self.ert.getEnsembleSize(),
            name=prior_name,
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior.id))
        prior_context = self.ert.ensemble_context(
            prior,
            np.array(self._simulation_arguments.active_realizations, dtype=bool),
            iteration=0,
        )

        self.ert.sample_prior(prior_context.sim_fs, prior_context.active_realizations)

        self._evaluate_and_postprocess(prior_context, evaluator_server_config)

        self.setPhaseName("Running ES update step")
        self.ert.runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.sim_fs
        )
        self.ert.runWorkflows(
            HookRuntime.PRE_UPDATE, self._storage, prior_context.sim_fs
        )
        states = [
            RealizationState.HAS_DATA,
            RealizationState.INITIALIZED,
        ]
        target_case_format = self._simulation_arguments.target_case
        posterior_context = self.ert.ensemble_context(
            self._storage.create_ensemble(
                self._experiment_id,
                ensemble_size=prior.ensemble_size,
                iteration=1,
                name=target_case_format,
                prior_ensemble=prior,
            ),
            prior_context.sim_fs.get_realization_mask_from_state(states),
            iteration=1,
        )

        try:
            self.facade.smoother_update(
                prior_context.sim_fs,
                posterior_context.sim_fs,
                prior_context.run_id,  # type: ignore
            )
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of experiment failed with the following error: {e}"
            ) from e

        self.ert.runWorkflows(
            HookRuntime.POST_UPDATE, self._storage, posterior_context.sim_fs
        )

        self._evaluate_and_postprocess(posterior_context, evaluator_server_config)

        self.setPhase(2, "Experiment completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
