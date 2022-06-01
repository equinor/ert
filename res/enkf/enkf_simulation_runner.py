from functools import partial

from cwrap import BaseCClass

from res import _lib
from res.enkf.ert_run_context import ErtRunContext
from res.job_queue import JobQueue, JobQueueManager, RunStatusType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from res.enkf import EnKFMain


class EnkfSimulationRunner(BaseCClass):
    TYPE_NAME = "enkf_simulation_runner"

    def __init__(self, enkf_main: "EnKFMain") -> None:
        assert isinstance(enkf_main, BaseCClass)
        # enkf_main should be an EnKFMain, get the _RealEnKFMain object
        real_enkf_main = enkf_main.parent()
        super().__init__(
            real_enkf_main.from_param(real_enkf_main).value,
            parent=real_enkf_main,
            is_reference=True,
        )

    def _enkf_main(self) -> "EnKFMain":
        return self.parent()

    def runSimpleStep(self, job_queue: JobQueue, run_context: ErtRunContext) -> int:
        # run simplestep
        self._enkf_main().initRun(run_context)

        if run_context.get_step():
            self._enkf_main().ecl_config.assert_restart()

        # start queue
        self.start_queue(run_context, job_queue)

        # deactivate failed realizations
        totalOk = 0
        totalFailed = 0
        for index, run_arg in enumerate(run_context):
            if run_context.is_active(index):
                if run_arg.run_status in (
                    RunStatusType.JOB_LOAD_FAILURE,
                    RunStatusType.JOB_RUN_FAILURE,
                ):
                    run_context.deactivate_realization(index)
                    totalFailed += 1
                else:
                    totalOk += 1

        run_context.get_sim_fs().fsync()

        if totalFailed == 0:
            print(f"All {totalOk} active jobs complete and data loaded.")
        else:
            print(f"{totalFailed} active job(s) failed.")

        return totalOk

    def createRunPath(self, run_context: ErtRunContext) -> None:
        self._enkf_main().initRun(run_context)
        _lib.enkf_main.write_run_path(self, run_context)

    def runEnsembleExperiment(
        self, job_queue: JobQueue, run_context: ErtRunContext
    ) -> int:
        return self.runSimpleStep(job_queue, run_context)

    @staticmethod
    def runWorkflows(runtime: int, ert: "EnKFMain") -> None:
        hook_manager = ert.getHookManager()
        hook_manager.runWorkflows(runtime, ert)

    def start_queue(self, run_context: ErtRunContext, job_queue: JobQueue) -> None:
        max_runtime = self._enkf_main().analysisConfig().get_max_runtime()
        if max_runtime == 0:
            max_runtime = None

        done_callback_function = _lib.model_callbacks.forward_model_ok
        exit_callback_function = _lib.model_callbacks.forward_model_exit

        # submit jobs
        for index, run_arg in enumerate(run_context):
            if not run_context.is_active(index):
                continue
            job_queue.add_job_from_run_arg(
                run_arg,
                self._enkf_main().resConfig(),
                max_runtime,
                done_callback_function,
                exit_callback_function,
            )

        job_queue.submit_complete()
        queue_evaluators = None
        if (
            self._enkf_main().analysisConfig().get_stop_long_running()
            and self._enkf_main().analysisConfig().minimum_required_realizations > 0
        ):
            queue_evaluators = [
                partial(
                    job_queue.stop_long_running_jobs,
                    self._enkf_main().analysisConfig().minimum_required_realizations,
                )
            ]

        jqm = JobQueueManager(job_queue, queue_evaluators)
        jqm.execute_queue()
