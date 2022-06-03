import asyncio
import concurrent
from functools import partial
import logging
import uuid

from typing import Optional, Dict, Any

from res.enkf.enums import HookRuntime, RealizationStateEnum
from res.enkf import ErtRunContext, EnkfSimulationRunner, ErtAnalysisError
from res.enkf.enkf_main import EnKFMain, QueueConfig

from ert_shared.models import BaseRunModel, ErtRunError
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator import identifiers

from cloudevents.http import CloudEvent


experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
    ):
        super().__init__(simulation_arguments, ert, queue_config, phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        loop = asyncio.get_running_loop()
        threadpool = concurrent.futures.ThreadPoolExecutor()

        prior_context = await loop.run_in_executor(threadpool, self.create_context)

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
            prior_context.get_iter(),
        )

        self._checkMinimumActiveRealizations(prior_context)

        await loop.run_in_executor(
            threadpool,
            self.ert().getEnkfSimulationRunner().createRunPath,
            prior_context,
        )

        await self._run_hook(
            HookRuntime.PRE_SIMULATION, prior_context.get_iter(), loop, threadpool
        )

        # Push ensemble, parameters, observations to new storage
        ensemble_id = await loop.run_in_executor(threadpool, self._post_ensemble_data)

        # Evaluate
        experiment_logger.debug("evaluating")
        await self._evaluate(prior_context, evaluator_server_config)

        # Push simulation results to storage
        await loop.run_in_executor(threadpool, self._post_ensemble_results, ensemble_id)

        num_successful_realizations = self._state_machine.successful_realizations(
            prior_context.get_iter(),
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        await self._run_hook(
            HookRuntime.POST_SIMULATION, prior_context.get_iter(), loop, threadpool
        )

        await self._run_hook(
            HookRuntime.PRE_FIRST_UPDATE, prior_context.get_iter(), loop, threadpool
        )

        await self._run_hook(
            HookRuntime.PRE_UPDATE, prior_context.get_iter(), loop, threadpool
        )

        # Send ANALYSIS_STARTED
        await self.dispatch(
            CloudEvent(
                {
                    "type": identifiers.EVTYPE_EXPERIMENT_ANALYSIS_STARTED,
                    "source": f"/ert/experiment/{self.id_}",
                    "id": str(uuid.uuid1()),
                }
            ),
            prior_context.get_iter(),
        )

        experiment_logger.debug("running update...")
        es_update = self.ert().getESUpdate()
        try:
            await loop.run_in_executor(
                threadpool, es_update.smootherUpdate, prior_context
            )
        except ErtAnalysisError as e:
            experiment_logger.exception("analysis failed")
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
                prior_context.get_iter(),
            )
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        experiment_logger.debug("update complete")

        # Send ANALYSIS_ENDED
        await self.dispatch(
            CloudEvent(
                {
                    "type": identifiers.EVTYPE_EXPERIMENT_ANALYSIS_ENDED,
                    "source": f"/ert/experiment/{self.id_}",
                    "id": str(uuid.uuid1()),
                }
            ),
            prior_context.get_iter(),
        )

        await self._run_hook(
            HookRuntime.POST_UPDATE, prior_context.get_iter(), loop, threadpool
        )

        # Create an update object in storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = await loop.run_in_executor(
            threadpool, self._post_update_data, ensemble_id, analysis_module_name
        )

        self.ert().getEnkfFsManager().switchFileSystem(prior_context.get_target_fs())

        experiment_logger.debug("create context for iter 1")
        rerun_context = await loop.run_in_executor(
            threadpool, partial(self.create_context, prior_context=prior_context)
        )

        await loop.run_in_executor(
            threadpool,
            self.ert().getEnkfSimulationRunner().createRunPath,
            rerun_context,
        )

        await self._run_hook(
            HookRuntime.PRE_SIMULATION, rerun_context.get_iter(), loop, threadpool
        )

        # Push ensemble, parameters, observations to new storage
        ensemble_id = await loop.run_in_executor(
            threadpool, self._post_ensemble_data, update_id
        )

        # Evaluate
        experiment_logger.debug("evaluating for iter 1")
        await self._evaluate(rerun_context, evaluator_server_config)

        num_successful_realizations = self._state_machine.successful_realizations(
            rerun_context.get_iter(),
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        await self._run_hook(
            HookRuntime.POST_SIMULATION, rerun_context.get_iter(), loop, threadpool
        )

        # Push simulation results to storage
        await loop.run_in_executor(threadpool, self._post_ensemble_results, ensemble_id)

        # Send EXPERIMENT_COMPLETED
        await self.dispatch(
            CloudEvent(
                {
                    "type": identifiers.EVTYPE_EXPERIMENT_SUCCEEDED,
                    "source": f"/ert/experiment/{self.id_}",
                    "id": str(uuid.uuid1()),
                },
            ),
            rerun_context.get_iter(),
        )
        experiment_logger.debug("experiment done")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> ErtRunContext:
        prior_context = self.create_context()

        self._checkMinimumActiveRealizations(prior_context)
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().getEnkfSimulationRunner().createRunPath(prior_context)

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())

        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data()

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            prior_context, evaluator_server_config
        )

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        self.setPhaseName("Analyzing...")
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_FIRST_UPDATE, ert=self.ert())
        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_UPDATE, ert=self.ert())
        es_update = self.ert().getESUpdate()
        try:
            es_update.smootherUpdate(prior_context)
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_UPDATE, ert=self.ert())

        # Create an update object in storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(ensemble_id, analysis_module_name)

        self.setPhase(1, "Running simulations...")
        self.ert().getEnkfFsManager().switchFileSystem(prior_context.get_target_fs())

        self.setPhaseName("Pre processing...")

        rerun_context = self.create_context(prior_context=prior_context)

        self.ert().getEnkfSimulationRunner().createRunPath(rerun_context)

        EnkfSimulationRunner.runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())
        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(update_id=update_id)

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            rerun_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        EnkfSimulationRunner.runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.setPhase(2, "Simulations completed.")

        return prior_context

    def create_context(
        self, prior_context: Optional[ErtRunContext] = None
    ) -> ErtRunContext:

        fs_manager = self.ert().getEnkfFsManager()
        if prior_context is None:
            sim_fs = fs_manager.getCurrentFileSystem()
            target_fs = fs_manager.getFileSystem(
                self._simulation_arguments["target_case"]
            )
            itr = 0
            mask = self._simulation_arguments["active_realizations"]
        else:
            itr = 1
            sim_fs = prior_context.get_target_fs()
            target_fs = None
            mask = sim_fs.getStateMap().createMask(
                RealizationStateEnum.STATE_HAS_DATA
                | RealizationStateEnum.STATE_INITIALIZED
            )

        run_context = self.ert().create_ensemble_smoother_run_context(
            active_mask=mask,
            iteration=itr,
            target_filesystem=target_fs,
            source_filesystem=sim_fs,
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.get_iter()
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
