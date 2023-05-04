from __future__ import annotations

import asyncio
import concurrent
import logging
from typing import TYPE_CHECKING, Any, Dict
from uuid import UUID

import _ert_com_protocol
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import BaseRunModel
from ert.shared.models.base_run_model import ErtRunError
from ert.storage import EnsembleAccessor, StorageAccessor

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain, QueueConfig


experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


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

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        experiment_logger.debug("starting ensemble experiment")
        event = _ert_com_protocol.node_status_builder(
            status="EXPERIMENT_STARTED", experiment_id=str(self.id)
        )
        await self.dispatch(event)

        current_case = self._simulation_arguments["current_case"]
        if isinstance(current_case, UUID):
            ensemble = self._storage.get_ensemble(current_case)
        else:
            ensemble = self._storage.create_ensemble(
                self._experiment_id,
                name=current_case,
                ensemble_size=self._ert.getEnsembleSize(),
            )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prior_context = await loop.run_in_executor(
                executor,
                self.ert().ensemble_context,
                ensemble,
                self._simulation_arguments["active_realizations"],
                self._simulation_arguments.get("iter_num", 0),
            )

            def sample_and_create_run_path(
                ert: "EnKFMain", run_context: "RunContext"
            ) -> None:
                ert.sample_prior(
                    run_context.sim_fs,
                    run_context.active_realizations,
                )
                ert.createRunPath(run_context)

            experiment_logger.debug("creating runpaths")
            await loop.run_in_executor(
                executor,
                sample_and_create_run_path,
                self.ert(),
                prior_context,
            )

            await self._run_hook(
                HookRuntime.PRE_SIMULATION, prior_context.iteration, loop, executor
            )

            experiment_logger.debug("evaluating")
            await self._evaluate(prior_context, evaluator_server_config)

            num_successful_realizations = await self.successful_realizations(
                prior_context.iteration
            )

            num_successful_realizations += self._simulation_arguments.get(
                "prev_successful_realizations", 0
            )
            try:
                self.checkHaveSufficientRealizations(num_successful_realizations)
            except ErtRunError as e:
                event = _ert_com_protocol.node_status_builder(
                    status="EXPERIMENT_FAILED", experiment_id=str(self._experiment_id)
                )
                event.experiment.message = str(e)
                await self.dispatch(event)
                return

            await self._run_hook(
                HookRuntime.POST_SIMULATION, prior_context.iteration, loop, executor
            )

        event = _ert_com_protocol.node_status_builder(
            status="EXPERIMENT_SUCCEEDED", experiment_id=str(self.id)
        )
        await self.dispatch(event)

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

        self.setPhase(1, "Simulations completed.")  # done...

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
