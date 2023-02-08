import asyncio
import concurrent
import logging
from typing import Any, Dict

import _ert_com_protocol
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime
from ert._clib.state_map import STATE_INITIALIZED, STATE_LOAD_FAILURE, STATE_UNDEFINED
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import BaseRunModel
from ert.shared.models.base_run_model import ErtRunError

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class EnsembleExperiment(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
        id_: str,
    ):
        super().__init__(simulation_arguments, ert, queue_config, id_)

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        experiment_logger.debug("starting ensemble experiment")
        event = _ert_com_protocol.node_status_builder(
            status="EXPERIMENT_STARTED", experiment_id=self.id_
        )
        await self.dispatch(event)

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prior_context = await loop.run_in_executor(
                executor,
                self.ert().load_ensemble_context,
                self.ert().storage_manager.current_case.case_name,
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

            ensemble_id = await loop.run_in_executor(executor, self._post_ensemble_data)

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
                    status="EXPERIMENT_FAILED", experiment_id=self.id_
                )
                event.experiment.message = str(e)
                await self.dispatch(event)
                return

            await self._run_hook(
                HookRuntime.POST_SIMULATION, prior_context.iteration, loop, executor
            )

            # Push simulation results to storage
            await loop.run_in_executor(
                executor,
                self._post_ensemble_results,
                ensemble_id,
            )

        event = _ert_com_protocol.node_status_builder(
            status="EXPERIMENT_SUCCEEDED", experiment_id=self.id_
        )
        await self.dispatch(event)

    def runSimulations__(
        self,
        run_msg: str,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> RunContext:
        prior_context = self.ert().load_ensemble_context(
            self.ert().storage_manager.current_case.case_name,
            self._simulation_arguments["active_realizations"],
            iteration=self._simulation_arguments.get("iter_num", 0),
        )

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        if not prior_context.sim_fs.is_initalized:
            self.ert().sample_prior(
                prior_context.sim_fs,
                prior_context.active_realizations,
            )
        else:
            state_map = prior_context.sim_fs.getStateMap()
            for realization_nr in prior_context.active_realizations:
                if state_map[realization_nr] in [STATE_UNDEFINED, STATE_LOAD_FAILURE]:
                    state_map[realization_nr] = STATE_INITIALIZED
        self.ert().createRunPath(prior_context)

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(prior_context.sim_fs.case_name)

        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION)

        self.setPhaseName(run_msg, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            prior_context, evaluator_server_config
        )

        num_successful_realizations += self._simulation_arguments.get(
            "prev_successful_realizations", 0
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION)

        # Push simulation results to storage
        self._post_ensemble_results(prior_context.sim_fs.case_name, ensemble_id)

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
