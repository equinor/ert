import asyncio
import concurrent
import logging
from functools import partial
from typing import Any, Dict, Optional

from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.analysis import ErtAnalysisError
from ert.ensemble_evaluator import identifiers
from ert.shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert.shared.models import BaseRunModel, ErtRunError

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class EnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
        id_: str,
    ):
        super().__init__(simulation_arguments, ert, queue_config, id_, phase_count=2)
        self.support_restart = False

    def setAnalysisModule(self, module_name: str) -> None:
        module_load_success = self.ert().analysisConfig().selectModule(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        loop = asyncio.get_running_loop()
        threadpool = concurrent.futures.ThreadPoolExecutor()

        prior_context = await loop.run_in_executor(threadpool, self.create_context)

        experiment_logger.debug("starting ensemble smoother experiment")
        await self._dispatch_ee(
            identifiers.EVTYPE_EXPERIMENT_STARTED, iteration=prior_context.iteration
        )

        self._checkMinimumActiveRealizations(prior_context)

        await loop.run_in_executor(
            threadpool,
            self.ert().createRunPath,
            prior_context,
        )

        await self._run_hook(
            HookRuntime.PRE_SIMULATION, prior_context.iteration, loop, threadpool
        )

        # Post ensemble, parameters, observations to new storage
        # (nb: this calls self.setPhaseName())
        ensemble_id = await loop.run_in_executor(threadpool, self._post_ensemble_data)

        experiment_logger.debug("evaluating")
        await self._evaluate(prior_context, evaluator_server_config)

        # Push simulation results to storage
        await loop.run_in_executor(threadpool, self._post_ensemble_results, ensemble_id)

        num_successful_realizations = await self.successful_realizations(
            prior_context.iteration
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        await self._run_hook(
            HookRuntime.POST_SIMULATION, prior_context.iteration, loop, threadpool
        )

        await self._run_hook(
            HookRuntime.PRE_FIRST_UPDATE, prior_context.iteration, loop, threadpool
        )

        await self._run_hook(
            HookRuntime.PRE_UPDATE, prior_context.iteration, loop, threadpool
        )

        await self._dispatch_ee(
            identifiers.EVTYPE_EXPERIMENT_ANALYSIS_STARTED,
            iteration=prior_context.iteration,
        )

        experiment_logger.debug("running update...")
        try:
            await loop.run_in_executor(
                threadpool, self.facade.smoother_update, prior_context
            )
        except ErtAnalysisError as e:
            experiment_logger.exception("analysis failed")
            await self._dispatch_ee(
                identifiers.EVTYPE_EXPERIMENT_FAILED,
                data={
                    "error": str(e),
                },
                iteration=prior_context.iteration,
            )
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        experiment_logger.debug("update complete")

        await self._dispatch_ee(
            identifiers.EVTYPE_EXPERIMENT_ANALYSIS_ENDED,
            iteration=prior_context.iteration,
        )

        await self._run_hook(
            HookRuntime.POST_UPDATE, prior_context.iteration, loop, threadpool
        )

        # Create an update object in storage
        update_id = await loop.run_in_executor(
            threadpool,
            self._post_update_data,
            ensemble_id,
            self.ert().analysisConfig().activeModuleName(),
        )

        self.ert().getEnkfFsManager().switchFileSystem(prior_context.target_fs)

        experiment_logger.debug("creating context for iter 1")
        rerun_context = await loop.run_in_executor(
            threadpool, partial(self.create_context, prior_context=prior_context)
        )

        await loop.run_in_executor(
            threadpool,
            self.ert().createRunPath,
            rerun_context,
        )

        await self._run_hook(
            HookRuntime.PRE_SIMULATION, rerun_context.iteration, loop, threadpool
        )

        # Push ensemble, parameters, observations to new storage
        ensemble_id = await loop.run_in_executor(
            threadpool, self._post_ensemble_data, update_id
        )

        # Evaluate
        experiment_logger.debug("evaluating for iter 1")
        await self._evaluate(rerun_context, evaluator_server_config)

        num_successful_realizations = await self.successful_realizations(
            rerun_context.iteration,
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        await self._run_hook(
            HookRuntime.POST_SIMULATION, rerun_context.iteration, loop, threadpool
        )

        # Push simulation results to storage
        await loop.run_in_executor(threadpool, self._post_ensemble_results, ensemble_id)

        await self._dispatch_ee(
            identifiers.EVTYPE_EXPERIMENT_SUCCEEDED,
            iteration=rerun_context.iteration,
        )
        experiment_logger.debug("experiment done")

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        prior_context = self.create_context()

        self._checkMinimumActiveRealizations(prior_context)
        self.setPhase(0, "Running simulations...", indeterminate=False)

        # self.setAnalysisModule(arguments["analysis_module"])

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().createRunPath(prior_context)

        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())

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
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        self.setPhaseName("Analyzing...")
        self.ert().runWorkflows(HookRuntime.PRE_FIRST_UPDATE, ert=self.ert())
        self.ert().runWorkflows(HookRuntime.PRE_UPDATE, ert=self.ert())
        try:
            self.facade.smoother_update(prior_context)
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        self.ert().runWorkflows(HookRuntime.POST_UPDATE, ert=self.ert())

        # Create an update object in storage
        analysis_module_name = self.ert().analysisConfig().activeModuleName()
        update_id = self._post_update_data(ensemble_id, analysis_module_name)

        self.setPhase(1, "Running simulations...")
        self.ert().getEnkfFsManager().switchFileSystem(prior_context.target_fs)

        self.setPhaseName("Pre processing...")

        rerun_context = self.create_context(prior_context=prior_context)

        self.ert().createRunPath(rerun_context)

        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION, ert=self.ert())
        # Push ensemble, parameters, observations to new storage
        ensemble_id = self._post_ensemble_data(update_id=update_id)

        self.setPhaseName("Running forecast...", indeterminate=False)

        num_successful_realizations = self.run_ensemble_evaluator(
            rerun_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION, ert=self.ert())

        # Push simulation results to storage
        self._post_ensemble_results(ensemble_id)

        self.setPhase(2, "Simulations completed.")

        return prior_context

    def create_context(self, prior_context: Optional[RunContext] = None) -> RunContext:
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
            sim_fs = prior_context.target_fs
            target_fs = None
            initialized_and_has_data: RealizationStateEnum = (
                RealizationStateEnum.STATE_HAS_DATA  # type: ignore
                | RealizationStateEnum.STATE_INITIALIZED
            )
            mask = sim_fs.getStateMap().createMask(initialized_and_has_data)

        run_context = self.ert().create_ensemble_smoother_run_context(
            active_mask=mask,
            iteration=itr,
            target_filesystem=target_fs,
            source_filesystem=sim_fs,
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.iteration
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Ensemble smoother"
