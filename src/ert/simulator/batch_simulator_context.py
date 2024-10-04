from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from _ert.threading import ErtThread
from ert.config import HookRuntime
from ert.enkf_main import create_run_path
from ert.ensemble_evaluator import Realization
from ert.runpaths import Runpaths
from ert.scheduler import JobState, Scheduler, create_driver
from ert.workflow_runner import WorkflowRunner

from ..run_arg import RunArg, create_run_arguments
from .forward_model_status import ForwardModelStatus

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from ert.config import ErtConfig
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)

Status = namedtuple("Status", "waiting pending running complete failed")


class DeprecatedJobStatus(Enum):
    # This value is used in external query routines - for jobs which are
    # (currently) not active.
    NOT_ACTIVE = auto()
    WAITING = auto()  # A node which is waiting in the internal queue.
    # Internal status: It has has been submitted - the next status update will
    # (should) place it as pending or running.
    SUBMITTED = auto()
    # A node which is pending - a status returned by the external system. I.e LSF
    PENDING = auto()
    RUNNING = auto()  # The job is running
    # The job is done - but we have not yet checked if the target file is
    # produced
    DONE = auto()
    # The job has exited - check attempts to determine if we retry or go to
    # complete_fail
    EXIT = auto()
    # The job has been killed, following a  DO_KILL - can restart.
    IS_KILLED = auto()
    # The the job should be killed, either due to user request, or automated
    # measures - the job can NOT be restarted..
    DO_KILL = auto()
    SUCCESS = auto()
    STATUS_FAILURE = auto()
    FAILED = auto()
    DO_KILL_NODE_FAILURE = auto()
    UNKNOWN = auto()


def _slug(entity: str) -> str:
    entity = " ".join(str(entity).split())
    return "".join([x if x.isalnum() else "_" for x in entity.strip()])


def _run_forward_model(
    ert_config: "ErtConfig",
    scheduler: Scheduler,
    run_args: List[RunArg],
) -> None:
    # run simplestep
    asyncio.run(_submit_and_run_jobqueue(ert_config, scheduler, run_args))


async def _submit_and_run_jobqueue(
    ert_config: "ErtConfig",
    scheduler: Scheduler,
    run_args: List[RunArg],
) -> None:
    max_runtime: Optional[int] = ert_config.analysis_config.max_runtime
    if max_runtime == 0:
        max_runtime = None
    for run_arg in run_args:
        if not run_arg.active:
            continue
        realization = Realization(
            iens=run_arg.iens,
            fm_steps=[],
            active=True,
            max_runtime=max_runtime,
            run_arg=run_arg,
            num_cpu=ert_config.preferred_num_cpu,
            job_script=ert_config.queue_config.job_script,
            realization_memory=ert_config.queue_config.realization_memory,
        )
        scheduler.set_realization(realization)

    required_realizations = 0
    if ert_config.queue_config.stop_long_running:
        required_realizations = ert_config.analysis_config.minimum_required_realizations
    with contextlib.suppress(asyncio.CancelledError):
        await scheduler.execute(required_realizations)


@dataclass
class BatchContext:
    result_keys: "Iterable[str]"
    ert_config: "ErtConfig"
    ensemble: Ensemble
    mask: npt.NDArray[np.bool_]
    itr: int
    case_data: List[Tuple[Any, Any]]

    def __post_init__(self) -> None:
        """
        Handle which can be used to query status and results for batch simulation.
        """
        ert_config = self.ert_config
        driver = create_driver(ert_config.queue_config)
        self._scheduler = Scheduler(
            driver, max_running=self.ert_config.queue_config.max_running
        )
        # fill in the missing geo_id data
        global_substitutions = self.ert_config.substitution_list
        global_substitutions["<CASE_NAME>"] = _slug(self.ensemble.name)
        for sim_id, (geo_id, _) in enumerate(self.case_data):
            if self.mask[sim_id]:
                global_substitutions[f"<GEO_ID_{sim_id}_{self.itr}>"] = str(geo_id)

        run_paths = Runpaths(
            jobname_format=ert_config.model_config.jobname_format_string,
            runpath_format=ert_config.model_config.runpath_format_string,
            filename=str(ert_config.runpath_file),
            substitution_list=global_substitutions,
            eclbase=ert_config.ensemble_config.eclbase,
        )
        self.run_args = create_run_arguments(
            run_paths,
            self.mask,
            ensemble=self.ensemble,
        )
        context_env = {
            "_ERT_EXPERIMENT_ID": str(self.ensemble.experiment_id),
            "_ERT_ENSEMBLE_ID": str(self.ensemble.id),
            "_ERT_SIMULATION_MODE": "batch_simulation",
        }
        create_run_path(
            self.run_args,
            self.ensemble,
            ert_config,
            run_paths,
            context_env,
        )
        for workflow in ert_config.hooked_workflows[HookRuntime.PRE_SIMULATION]:
            WorkflowRunner(workflow, None, self.ensemble).run_blocking()
        self._sim_thread = self._run_simulations_simple_step()

        # Wait until the queue is active before we finish the creation
        # to ensure sane job status while running
        while self.running() and not self._scheduler.is_active():
            time.sleep(0.1)

    def __len__(self) -> int:
        return len(self.mask)

    def get_ensemble(self) -> Ensemble:
        return self.ensemble

    def _run_simulations_simple_step(self) -> Thread:
        sim_thread = ErtThread(
            target=lambda: _run_forward_model(
                self.ert_config, self._scheduler, self.run_args
            )
        )
        sim_thread.start()
        return sim_thread

    def join(self) -> None:
        """
        Will block until the simulation is complete.
        """
        while self.running():
            time.sleep(1)

    def running(self) -> bool:
        return self._sim_thread.is_alive() or self._scheduler.is_active()

    @property
    def status(self) -> Status:
        """
        Will return the state of the simulations.

        NB: Killed realizations are not reported.
        """
        states = self._scheduler.count_states()
        return Status(
            running=states[JobState.RUNNING],
            waiting=states[JobState.WAITING],
            pending=states[JobState.PENDING],
            complete=states[JobState.COMPLETED],
            failed=states[JobState.FAILED],
        )

    def results(self) -> List[Optional[Dict[str, "npt.NDArray[np.float64]"]]]:
        """Will return the results of the simulations.

        Observe that this function will raise RuntimeError if the simulations
        have not been completed. To be certain that the simulations have
        completed you can call the join() method which will block until all
        simulations have completed.

        The function will return all the results which were configured with the
        @results when the simulator was created. The results will come as a
        list of dictionaries of arrays of double values, i.e. if the @results
        argument was:

             results = ["CMODE", "order"]

        when the simulator was created the results will be returned as:


          [ {"CMODE" : [1,2,3], "order" : [1,1,3]},
            {"CMODE" : [1,4,1], "order" : [0,7,8]},
            None,
            {"CMODE" : [6,1,0], "order" : [0,0,8]} ]

        For a simulation which consist of a total of four simulations, where the
        None value indicates that the simulator was unable to compute a request.
        The order of the list corresponds to case_data provided in the start
        call.

        """
        if self.running():
            raise RuntimeError(
                "Simulations are still running - need to wait before getting results"
            )

        res: List[Optional[Dict[str, "npt.NDArray[np.float64]"]]] = []
        for sim_id in range(len(self)):
            if self.get_job_state(iens=sim_id) != JobState.COMPLETED:
                logger.error(f"Simulation {sim_id} failed.")
                res.append(None)
                continue
            d = {}
            for key in self.result_keys:
                data = self.ensemble.load_responses(key, (sim_id,))
                d[key] = data["values"].values.flatten()
            res.append(d)

        return res

    def job_status(self, iens: int) -> Optional["DeprecatedJobStatus"]:
        """Will query the queue system for the status of the job."""
        state_to_legacy = {
            JobState.WAITING: DeprecatedJobStatus.WAITING,
            JobState.SUBMITTING: DeprecatedJobStatus.SUBMITTED,
            JobState.PENDING: DeprecatedJobStatus.PENDING,
            JobState.RUNNING: DeprecatedJobStatus.RUNNING,
            JobState.ABORTING: DeprecatedJobStatus.DO_KILL,
            JobState.COMPLETED: DeprecatedJobStatus.SUCCESS,
            JobState.FAILED: DeprecatedJobStatus.FAILED,
            JobState.ABORTED: DeprecatedJobStatus.IS_KILLED,
        }
        return state_to_legacy[self._scheduler._jobs[iens].state]

    def is_job_completed(self, iens: int) -> bool:
        return self.get_job_state(iens) == JobState.COMPLETED

    def has_job_failed(self, iens: int) -> bool:
        return self.get_job_state(iens) == JobState.FAILED

    def is_job_waiting(self, iens: int) -> bool:
        return self.get_job_state(iens) in [
            JobState.WAITING,
            JobState.SUBMITTING,
            JobState.PENDING,
        ]

    def get_job_state(self, iens: int) -> Optional["JobState"]:
        """Will query the queue system for the status of the job."""
        return self._scheduler._jobs[iens].state

    def job_progress(self, iens: int) -> Optional[ForwardModelStatus]:
        """Will return a detailed progress of the job.
        The progress report is obtained by reading a file from the filesystem,
        that file is typically created by another process running on another
        machine, and reading might fail due to NFS issues, simultanoues write
        and so on. If loading valid json fails the function will sleep 0.10
        seconds and retry - eventually giving up and returning None. Also for
        jobs which have not yet started the method will return None.
        When the method succeeds in reading the progress file from the file
        system the return value will be an object with properties like this:
            progress.start_time
            progress.end_time
            progress.run_id
            progress.jobs = [
                (job1.name, job1.start_time, job1.end_time, job1.status, job1.error_msg),
                (job2.name, job2.start_time, job2.end_time, job2.status, job2.error_msg),
                (jobN.name, jobN.start_time, jobN.end_time, jobN.status, jobN.error_msg)
            ]
        """
        try:
            run_arg = self.run_args[iens]
        except IndexError as e:
            raise KeyError(e) from e

        if (
            iens not in self._scheduler._jobs
            or self._scheduler._jobs[iens].state == JobState.WAITING
        ):
            return None
        return ForwardModelStatus.load(run_arg.runpath)

    def stop(self) -> None:
        self._scheduler.kill_all_jobs()
        self._sim_thread.join()

    def run_path(self, iens: int) -> str:
        return self.run_args[iens].runpath
