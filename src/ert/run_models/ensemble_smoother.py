from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict
from uuid import UUID

from ert.analysis import ErtAnalysisError
from ert.config import HookRuntime
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.realization_state import RealizationState
from ert.run_context import RunContext
from ert.storage import StorageAccessor

from .base_run_model import BaseRunModel, ErtRunError

if TYPE_CHECKING:
    from ert.config import QueueConfig
    from ert.enkf_main import EnKFMain

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


# pylint: disable=too-many-arguments
class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        experiment_id: UUID,
    ):
        super().__init__(
            simulation_arguments,
            ert,
            storage,
            queue_config,
            experiment_id,
            phase_count=2,
        )
        self._current_case_name: str = simulation_arguments["current_case"]
        self.support_restart = False

    async def run(self, _: EvaluatorServerConfig) -> None:
        raise NotImplementedError()

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().select_module(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        prior_name = self._simulation_arguments["current_case"]
        prior_fs = self._storage.create_ensemble(
            self._experiment_id,
            name=prior_name,
            ensemble_size=self._ert.getEnsembleSize(),
        )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(prior_fs.id))
        prior_context = self._ert.ensemble_context(
            prior_fs,
            self._simulation_arguments["active_realizations"],
            iteration=0,
        )

        self._checkMinimumActiveRealizations(len(prior_context.active_realizations))
        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().sample_prior(prior_context.sim_fs, prior_context.active_realizations)
        self.ert().createRunPath(prior_context)

        self.ert().runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, prior_context.sim_fs
        )

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            prior_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, prior_context.sim_fs
        )

        self.setPhaseName("Analyzing...")
        self.ert().runWorkflows(
            HookRuntime.PRE_FIRST_UPDATE, self._storage, prior_context.sim_fs
        )
        self.ert().runWorkflows(
            HookRuntime.PRE_UPDATE, self._storage, prior_context.sim_fs
        )
        states = [
            RealizationState.HAS_DATA,
            RealizationState.INITIALIZED,
        ]
        target_case_format = self._simulation_arguments["target_case"]
        posterior_context = self.ert().ensemble_context(
            self._storage.create_ensemble(
                self._experiment_id,
                ensemble_size=self._ert.getEnsembleSize(),
                iteration=1,
                name=target_case_format,
                prior_ensemble=prior_fs,
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
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        self.ert().runWorkflows(
            HookRuntime.POST_UPDATE, self._storage, posterior_context.sim_fs
        )

        self.setPhase(1, "Running simulations...")

        self.setPhaseName("Pre processing...")

        self.ert().createRunPath(posterior_context)

        self.ert().runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, posterior_context.sim_fs
        )

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            posterior_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, posterior_context.sim_fs
        )

        self.setPhase(2, "Simulations completed.")

        return posterior_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
