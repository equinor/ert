from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict
from uuid import UUID

from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.storage import EnsembleAccessor, StorageAccessor

from .base_run_model import BaseRunModel

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain, QueueConfig


# pylint: disable=too-many-arguments
class EnsembleExperiment(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        storage: StorageAccessor,
        queue_config: QueueConfig,
        id_: UUID,
    ):
        super().__init__(simulation_arguments, ert, storage, queue_config, id_)

    def runSimulations__(
        self,
        run_msg: str,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> RunContext:
        current_case = self._simulation_arguments["current_case"]
        try:
            ensemble = self._storage.get_ensemble_by_name(current_case)
            assert isinstance(ensemble, EnsembleAccessor)
        except KeyError:
            ensemble = self._storage.create_ensemble(
                self._experiment_id,
                name=current_case,
                ensemble_size=self._ert.getEnsembleSize(),
                iteration=self._simulation_arguments.get("item_num", 0),
            )
        self.set_env_key("_ERT_ENSEMBLE_ID", str(ensemble.id))

        prior_context = self._ert.ensemble_context(
            ensemble,
            self._simulation_arguments["active_realizations"],
            iteration=self._simulation_arguments.get("iter_num", 0),
        )

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        if not prior_context.sim_fs.realizations_initialized(
            prior_context.active_realizations
        ):
            self.ert().sample_prior(
                prior_context.sim_fs,
                prior_context.active_realizations,
            )
        else:
            state_map = prior_context.sim_fs.state_map
            for realization_nr in prior_context.active_realizations:
                if state_map[realization_nr] in [
                    RealizationStateEnum.STATE_UNDEFINED,
                    RealizationStateEnum.STATE_LOAD_FAILURE,
                ]:
                    state_map[realization_nr] = RealizationStateEnum.STATE_INITIALIZED
        self.ert().createRunPath(prior_context)

        self.ert().runWorkflows(
            HookRuntime.PRE_SIMULATION, self._storage, prior_context.sim_fs
        )

        self.setPhaseName(run_msg, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            prior_context, evaluator_server_config
        )

        num_successful_realizations += self._simulation_arguments.get(
            "prev_successful_realizations", 0
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(
            HookRuntime.POST_SIMULATION, self._storage, prior_context.sim_fs
        )

        self.setPhase(1, "Simulations completed.")

        return prior_context

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        return self.runSimulations__(
            "Running ensemble experiment...", evaluator_server_config
        )

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"
