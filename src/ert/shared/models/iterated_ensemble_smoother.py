import asyncio
import concurrent
import logging
from typing import Any, Dict, Optional

from iterative_ensemble_smoother import IterativeEnsembleSmoother

import _ert_com_protocol
from ert._c_wrappers.analysis.analysis_module import AnalysisModule
from ert._c_wrappers.enkf import RunContext
from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.enkf_main import EnKFMain, QueueConfig
from ert._c_wrappers.enkf.enums import HookRuntime, RealizationStateEnum
from ert.analysis import ErtAnalysisError
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.shared.models import BaseRunModel, ErtRunError

experiment_logger = logging.getLogger("ert.experiment_server.ensemble_experiment")


class IteratedEnsembleSmoother(BaseRunModel):
    def __init__(
        self,
        simulation_arguments: Dict[str, Any],
        ert: EnKFMain,
        queue_config: QueueConfig,
        id_: str,
    ):
        super().__init__(simulation_arguments, ert, queue_config, id_, phase_count=2)
        self.support_restart = False
        self._w_container = IterativeEnsembleSmoother(
            len(simulation_arguments["active_realizations"])
        )

    def setAnalysisModule(self, module_name: str) -> AnalysisModule:
        module_load_success = self.ert().analysisConfig().select_module(module_name)

        if not module_load_success:
            raise ErtRunError(f"Unable to load analysis module '{module_name}'!")

        return self.ert().analysisConfig().get_module(module_name)

    def _runAndPostProcess(
        self,
        run_context: RunContext,
        evaluator_server_config: EvaluatorServerConfig,
        update_id: Optional[str] = None,
    ) -> str:
        phase_msg = (
            f"Running iteration {run_context.iteration} of "
            f"{self.phaseCount() - 1} simulation iterations..."
        )
        self.setPhase(run_context.iteration, phase_msg, indeterminate=False)

        self.setPhaseName("Pre processing...", indeterminate=True)
        self.ert().createRunPath(run_context)
        self.ert().runWorkflows(HookRuntime.PRE_SIMULATION)
        # create ensemble
        ensemble_id = self._post_ensemble_data(update_id=update_id)
        self.setPhaseName("Running forecast...", indeterminate=False)
        num_successful_realizations = self.run_ensemble_evaluator(
            run_context, evaluator_server_config
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        self.setPhaseName("Post processing...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_SIMULATION)
        self._post_ensemble_results(ensemble_id)
        return ensemble_id

    def createTargetCaseFileSystem(self, phase: int, target_case_format: str) -> EnkfFs:
        target_fs = self.ert().getFileSystem(target_case_format % phase)
        return target_fs

    def analyzeStep(self, run_context: RunContext, ensemble_id: str) -> str:
        self.setPhaseName("Analyzing...", indeterminate=True)

        self.setPhaseName("Pre processing update...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.PRE_UPDATE)

        try:
            self.facade.iterative_smoother_update(run_context, self._w_container)
        except ErtAnalysisError as e:
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().active_module_name()
        update_id = self._post_update_data(
            parent_ensemble_id=ensemble_id, algorithm=analysis_module_name
        )

        self.setPhaseName("Post processing update...", indeterminate=True)
        self.ert().runWorkflows(HookRuntime.POST_UPDATE)
        return update_id

    def runSimulations(
        self, evaluator_server_config: EvaluatorServerConfig
    ) -> RunContext:
        phase_count = self.facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        target_case_format = self._simulation_arguments["target_case"]
        run_context = self.create_context(0)

        self.ert().analysisConfig().set_case_format(target_case_format)

        ensemble_id = self._runAndPostProcess(run_context, evaluator_server_config)

        analysis_config = self.ert().analysisConfig()
        num_retries_per_iteration = analysis_config.num_retries_per_iter
        num_retries = 0
        current_iter = 0

        while (
            current_iter < self.facade.get_number_of_iterations()
            and num_retries < num_retries_per_iteration
        ):
            pre_analysis_iter_num = self._w_container.iteration_nr - 1
            # We run the PRE_FIRST_UPDATE hook here because the current_iter is
            # explicitly available, versus in the run_context inside analyzeStep
            if current_iter == 0:
                self.ert().runWorkflows(HookRuntime.PRE_FIRST_UPDATE)
            update_id = self.analyzeStep(run_context, ensemble_id)
            current_iter = self._w_container.iteration_nr - 1

            analysis_success = current_iter > pre_analysis_iter_num
            if analysis_success:
                run_context = self.create_context(
                    current_iter, prior_context=run_context
                )
                ensemble_id = self._runAndPostProcess(
                    run_context, evaluator_server_config, update_id
                )
                num_retries = 0
            else:
                run_context = self.create_context(
                    current_iter, prior_context=run_context, rerun=True
                )
                ensemble_id = self._runAndPostProcess(
                    run_context, evaluator_server_config, update_id
                )
                num_retries += 1

        if current_iter == (phase_count - 1):
            self.setPhase(phase_count, "Simulations completed.")
        else:
            raise ErtRunError(
                (
                    "Iterated ensemble smoother stopped: "
                    "maximum number of iteration retries "
                    f"({num_retries_per_iteration} retries) reached "
                    f"for iteration {current_iter}"
                )
            )

        return run_context

    def create_context(
        self,
        itr: int,
        prior_context: Optional[RunContext] = None,
        rerun: bool = False,
    ) -> RunContext:

        target_case_format = self._simulation_arguments["target_case"]

        sim_fs = self.createTargetCaseFileSystem(itr, target_case_format)

        if prior_context is None:
            mask = self._simulation_arguments["active_realizations"]
        else:
            state: RealizationStateEnum = (
                RealizationStateEnum.STATE_HAS_DATA  # type: ignore
                | RealizationStateEnum.STATE_INITIALIZED
            )
            mask = sim_fs.getStateMap().createMask(state)

        if rerun or itr >= self.facade.get_number_of_iterations():
            target_fs = None
        else:
            target_fs = self.createTargetCaseFileSystem(itr + 1, target_case_format)

        run_context = self.ert().create_ensemble_smoother_run_context(
            source_filesystem=sim_fs,
            target_filesystem=target_fs,
            active_mask=mask,
            iteration=itr,
        )
        self._run_context = run_context
        self._last_run_iteration = run_context.iteration
        self.ert().getEnkfFsManager().switchFileSystem(sim_fs.case_name)
        return run_context

    @classmethod
    def name(cls) -> str:
        return "Iterated ensemble smoother"

    async def run(self, evaluator_server_config: EvaluatorServerConfig) -> None:
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor()

        phase_count = self.facade.get_number_of_iterations() + 1
        self.setPhaseCount(phase_count)

        target_case_format = self._simulation_arguments["target_case"]
        run_context = await loop.run_in_executor(executor, self.create_context, 0)

        self.ert().analysisConfig().set_case_format(target_case_format)

        ensemble_id = await self._run_and_post_process(
            loop, executor, run_context, evaluator_server_config
        )

        analysis_config = self.ert().analysisConfig()
        num_retries_per_iteration = analysis_config.num_retries_per_iter
        num_retries = 0
        current_iter = 0

        while (
            current_iter < self.facade.get_number_of_iterations()
            and num_retries < num_retries_per_iteration
        ):
            pre_analysis_iter_num = self._w_container.iteration_nr - 1
            # We run the PRE_FIRST_UPDATE hook here because the current_iter is
            # explicitly available, versus in the run_context inside analyzeStep
            if current_iter == 0:
                await self._run_hook(
                    HookRuntime.PRE_FIRST_UPDATE,
                    current_iter,
                    loop,
                    executor,
                )
            update_id = await self.analyze_step(
                loop, executor, run_context, ensemble_id
            )
            current_iter = self._w_container.iteration_nr - 1

            analysis_success = current_iter > pre_analysis_iter_num
            if analysis_success:
                run_context = await loop.run_in_executor(
                    executor, self.create_context, current_iter, run_context
                )

                ensemble_id = await self._run_and_post_process(
                    loop, executor, run_context, evaluator_server_config, update_id
                )
                num_retries = 0
            else:
                run_context = await loop.run_in_executor(
                    executor, self.create_context, current_iter, run_context, True
                )

                ensemble_id = await self._run_and_post_process(
                    loop, executor, run_context, evaluator_server_config, update_id
                )

                num_retries += 1

        if current_iter == (phase_count - 1):
            self.setPhase(current_iter, "Simulations completed.")
        else:
            raise ErtRunError(
                (
                    "Iterated ensemble smoother stopped: "
                    "maximum number of iteration retries "
                    f"({num_retries_per_iteration} retries) reached "
                    f"for iteration {current_iter}"
                )
            )

    async def _run_and_post_process(
        self,
        loop: asyncio.AbstractEventLoop,
        executor: concurrent.futures.Executor,
        run_context: RunContext,
        evaluator_server_config: EvaluatorServerConfig,
        update_id: Optional[str] = None,
    ) -> str:
        await loop.run_in_executor(
            executor,
            self.ert().createRunPath,
            run_context,
        )

        await self._run_hook(
            HookRuntime.PRE_SIMULATION,
            run_context.iteration,
            loop,
            executor,
        )

        ensemble_id = await loop.run_in_executor(
            executor, self._post_ensemble_data, update_id
        )

        experiment_logger.debug("evaluating")
        await self._evaluate(run_context, evaluator_server_config)

        num_successful_realizations = await self.successful_realizations(
            run_context.iteration,
        )

        self.checkHaveSufficientRealizations(num_successful_realizations)

        await self._run_hook(
            HookRuntime.POST_SIMULATION,
            run_context.iteration,
            loop,
            executor,
        )

        self._post_ensemble_results(ensemble_id)

        await loop.run_in_executor(executor, self._post_ensemble_data, ensemble_id)

        return ensemble_id

    async def analyze_step(
        self,
        loop: asyncio.AbstractEventLoop,
        executor: concurrent.futures.Executor,
        run_context: RunContext,
        ensemble_id: str,
    ) -> str:
        await self._run_hook(
            HookRuntime.PRE_UPDATE,
            run_context.iteration,
            loop,
            executor,
        )

        try:
            await loop.run_in_executor(
                executor,
                self.facade.iterative_smoother_update,
                run_context,
                self._w_container,
            )
        except ErtAnalysisError as e:
            experiment_logger.exception("analysis failed")
            event = _ert_com_protocol.node_status_builder(
                status="EXPERIMENT_FAILED", experiment_id=self.id_
            )
            event.experiment.message = str(e)
            await self.dispatch(event)
            raise ErtRunError(
                f"Analysis of simulation failed with the following error: {e}"
            ) from e

        # Push update data to new storage
        analysis_module_name = self.ert().analysisConfig().active_module_name()
        update_id = await loop.run_in_executor(
            executor, self._post_update_data, ensemble_id, analysis_module_name
        )

        await self._run_hook(
            HookRuntime.POST_UPDATE, run_context.iteration, loop, executor
        )
        return update_id
