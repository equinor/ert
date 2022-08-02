import logging
import concurrent
import asyncio
from ert_shared.models.base_run_model import ErtRunError
from res.enkf.enkf_main import EnKFMain, QueueConfig
from res.enkf.enums import HookRuntime
from res.enkf import RunContext, EnkfSimulationRunner
import uuid
from ert_shared.models import BaseRunModel
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from typing import Dict, Any
from ert.ensemble_evaluator import identifiers

from cloudevents.http import CloudEvent

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class EnsembleExperiment(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
    ):
        super().__init__(simulation_arguments, ert, queue_config)

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:

        # Send EXPERIMENT_STARTED
        experiment_logger.debug("starting ensemble experiment")
        await self.dispatch(
            CloudEvent(
                {
                    "type": identifiers.EVTYPE_EXPERIMENT_STARTED,
                    "source": f"/ert/experiment/{self.id_}",
                    "id": str(uuid.uuid1()),
                }
            ),
            0,
        )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            run_context = await loop.run_in_executor(executor, self.create_context)

            # Create runpaths
            experiment_logger.debug("creating runpaths")
            await loop.run_in_executor(
                executor,
                self.ert().getEnkfSimulationRunner().createRunPath,
                run_context,
            )

            ensemble_id = await loop.run_in_executor(executor, self._post_ensemble_data)

            await self._run_hook(
                HookRuntime.PRE_SIMULATION, run_context.iteration, loop, executor
            )

            # Evaluate
            experiment_logger.debug("evaluating")
            await self._evaluate(run_context, evaluator_server_config)

            num_successful_realizations = self._state_machine.successful_realizations(
                run_context.iteration
            )

            num_successful_realizations += self._simulation_arguments.get(
                "prev_successful_realizations", 0
            )
            try:
                self.checkHaveSufficientRealizations(num_successful_realizations)
            except ErtRunError as e:

                # Send EXPERIMENT_FAILED
                await self.dispatch(
                    CloudEvent(
                        {
                            "type": identifiers.EVTYPE_EXPERIMENT_FAILED,
                            "source": f"/ert/experiment/{self.id_}",
                            "id": str(uuid.uuid1()),
                        },
                        {
                            "error": str(e),
                        },
                    ),
                    run_context.iteration,
                )
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

        # Send EXPERIMENT_COMPLETED
        await self.dispatch(
            CloudEvent(
                {
                    "type": identifiers.EVTYPE_EXPERIMENT_SUCCEEDED,
                    "source": f"/ert/experiment/{self.id_}",
                    "id": str(uuid.uuid1()),
                },
            ),
            run_context.iteration,
        )

    def runSimulations__(
        self,
        run_msg: str,
        evaluator_server_config: EvaluatorServerConfig,
    ) -> RunContext:

        run_context = self.create_context()

        self.setPhase(0, "Running simulations...", indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(run_context)

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data()

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, self.ert())

        self.setPhaseName(run_msg, indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        num_successful_realizations += self._simulation_arguments.get(
            "prev_successful_realizations", 0
        )
        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, self.ert())

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
