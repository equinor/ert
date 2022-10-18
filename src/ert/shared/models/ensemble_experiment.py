import asyncio
import concurrent
import logging
from typing import Any, Dict

import _ert_com_protocol
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime
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
            run_context = await loop.run_in_executor(executor, self.create_context)

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
                run_context,
            )

            ensemble_id = await loop.run_in_executor(executor, self._post_ensemble_data)

            await self._run_hook(
                HookRuntime.PRE_SIMULATION, run_context.iteration, loop, executor
            )

            experiment_logger.debug("evaluating")
            await self._evaluate(run_context, evaluator_server_config)

            num_successful_realizations = await self.successful_realizations(
                run_context.iteration
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
                HookRuntime.POST_SIMULATION, run_context.iteration, loop, executor
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

        run_context = self.create_context()

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().sample_prior(
            run_context.sim_fs,
            run_context.active_realizations,
        )
        self.ert().createRunPath(run_context)

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data()

        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION)

        self.setPhaseName(run_msg, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        num_successful_realizations += self._simulation_arguments.get(
            "prev_successful_realizations", 0
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION)

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.setPhase(1, "Simulations completed.")  # done...

        return run_context

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        return self.runSimulations__(
            "Running ensemble experiment...", evaluator_server_config
        )

    def create_context(self) -> RunContext:
        itr = self._simulation_arguments.get("iter_num", 0)
        mask = self._simulation_arguments["active_realizations"]

        run_context = self.ert().create_ensemble_experiment_run_context(
            active_mask=mask,
            iteration=itr,
        )

        self._run_context = run_context
        self._last_run_iteration = run_context.iteration
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble experiment"
