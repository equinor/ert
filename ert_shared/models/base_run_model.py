from typing import Optional

from ert_shared.storage.extraction import (
    post_ensemble_data,
    post_ensemble_results,
    post_update_data,
)
from res.enkf.ert_run_context import ErtRunContext
from ert_shared.feature_toggling import FeatureToggling, feature_enabled
import logging
import time
import uuid
import asyncio

from ecl.util.util import BoolVector
from ert_shared import ERT
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.ensemble.builder import (
    create_ensemble_builder_from_legacy,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from res.enkf.enums.realization_state_enum import RealizationStateEnum
from res.job_queue import ForwardModelStatus, JobStatusType, RunStatusType
from res.util import ResLog

# A method decorated with the @job_queue decorator implements the following logic:
#
# 1. If self._job_queue is assigned a valid value the method is run normally.
# 2. If self._job_queue is None - the decorator argument is returned.


def job_queue(default):
    def job_queue_decorator(method):
        def dispatch(self, *args, **kwargs):

            if self._job_queue is None:
                return default
            else:
                return method(self, *args, **kwargs)

        return dispatch

    return job_queue_decorator


class ErtRunError(Exception):
    pass


class BaseRunModel(object):
    def __init__(self, queue_config, phase_count=1):
        super(BaseRunModel, self).__init__()
        self._phase = 0
        self._phase_count = phase_count
        self._phase_name = "Starting..."

        self._job_start_time = 0
        self._job_stop_time = 0
        self._indeterminate = False
        self._fail_message = ""
        self._failed = False
        self._queue_config = queue_config
        self._job_queue = None
        self.realization_progress = {}
        self.initial_realizations_mask = []
        self.completed_realizations_mask = []
        self.support_restart = True
        self._run_context = None
        self._last_run_iteration = -1
        self.reset()

    def ert(self):
        """@rtype: res.enkf.EnKFMain"""
        return ERT.ert

    @property
    def _ensemble_size(self):
        return self.initial_realizations_mask.count(True)

    def reset(self):
        self._failed = False
        self._phase = 0

    def start_simulations_thread(self, arguments):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.startSimulations(arguments=arguments)

    def startSimulations(self, arguments):
        try:
            self.initial_realizations_mask = arguments["active_realizations"]
            run_context = self.runSimulations(arguments)
            self.updateDetailedProgress()
            self.completed_realizations_mask = run_context.get_mask()
        except ErtRunError as e:
            self.updateDetailedProgress()
            self.completed_realizations_mask = BoolVector(default_value=False)
            self._failed = True
            self._fail_message = str(e)
            self._simulationEnded()
        except UserWarning as e:
            self.updateDetailedProgress()
            self._fail_message = str(e)
            self._simulationEnded()
        except Exception as e:
            self.updateDetailedProgress()
            self._failed = True
            self._fail_message = str(e)
            self._simulationEnded()
            raise

    def runSimulations(self, job_queue, run_context):
        raise NotImplementedError("Method must be implemented by inheritors!")

    def create_context(self, arguments):
        raise NotImplementedError("Method must be implemented by inheritors!")

    def teardown_context(self):
        # Used particularly to delete last active run_context to notify
        # fs_manager that storage is not being written to.
        self._run_context = None

    @job_queue(None)
    def killAllSimulations(self):
        if FeatureToggling.is_enabled("ensemble-evaluator"):
            raise NotImplementedError(
                "the ensemble evaluator does not implement killAllSimulations"
            )
        self._job_queue.kill_all_jobs()

    @job_queue(False)
    def userExitCalled(self):
        """@rtype: bool"""
        return self._job_queue.getUserExit()

    def phaseCount(self):
        """@rtype: int"""
        return self._phase_count

    def setPhaseCount(self, phase_count):
        self._phase_count = phase_count
        self.setPhase(0, "")

    def currentPhase(self):
        """@rtype: int"""
        return self._phase

    def setPhaseName(self, phase_name, indeterminate=None):
        self._phase_name = phase_name
        self.setIndeterminate(indeterminate)

    def getPhaseName(self):
        """@rtype: str"""
        return self._phase_name

    def setIndeterminate(self, indeterminate):
        if indeterminate is not None:
            self._indeterminate = indeterminate

    def isFinished(self):
        """@rtype: bool"""
        return self._phase == self._phase_count or self.hasRunFailed()

    def hasRunFailed(self):
        """@rtype: bool"""
        return self._failed

    def getFailMessage(self):
        """@rtype: str"""
        return self._fail_message

    def _simulationEnded(self):
        self._job_stop_time = int(time.time())

    def setPhase(self, phase, phase_name, indeterminate=None):
        self.setPhaseName(phase_name)
        if not 0 <= phase <= self._phase_count:
            raise ValueError(
                "Phase must be an integer from 0 to less than %d." % self._phase_count
            )

        self.setIndeterminate(indeterminate)

        if phase == 0:
            self._job_start_time = int(time.time())

        if phase == self._phase_count:
            self._simulationEnded()

        self._phase = phase

    def stop_time(self):
        return self._job_stop_time

    def start_time(self):
        return self._job_start_time

    def get_runtime(self):
        if self.stop_time() < self.start_time():
            return time.time() - self.start_time()
        else:
            return self.stop_time() - self.start_time()

    @job_queue(1)
    def getQueueSize(self):
        """@rtype: int"""
        queue_size = len(self._job_queue)

        if queue_size == 0:
            queue_size = 1

        return queue_size

    @job_queue({})
    def getQueueStatus(self):
        """@rtype: dict of (JobStatusType, int)"""
        queue_status = {}

        for job_number in range(len(self._job_queue)):
            status = self._job_queue.getJobStatus(job_number)

            if not status in queue_status:
                queue_status[status] = 0

            queue_status[status] += 1

        return queue_status

    @job_queue(False)
    def isQueueRunning(self):
        """@rtype: bool"""
        return self._job_queue.isRunning()

    @staticmethod
    def is_forward_model_finished(progress):
        return not (any((job.status != "Success" for job in progress)))

    def update_progress_for_index(self, iteration, idx, run_arg):
        try:
            # will throw if not yet submitted (is in a limbo state)
            queue_index = run_arg.getQueueIndex()
        except (ValueError, AttributeError):
            return

        status = None
        timed_out = False
        if self._job_queue:
            status = self._job_queue.getJobStatus(queue_index)
            timed_out = self._job_queue.did_job_time_out(queue_index)

        # Avoids reading from disk for jobs in these states since there's no
        # data anyway. If timed out, never exit here as that would prevent
        # propagation of the failure status.
        if (
            status
            in [
                JobStatusType.JOB_QUEUE_PENDING,
                JobStatusType.JOB_QUEUE_SUBMITTED,
                JobStatusType.JOB_QUEUE_WAITING,
            ]
            and not timed_out
        ):
            return

        fms = self.realization_progress[iteration].get(run_arg.iens, None)
        jobs = fms[0] if fms else None

        # Don't load from file if you are finished
        if not fms or not BaseRunModel.is_forward_model_finished(fms[0]):
            loaded = ForwardModelStatus.load(run_arg.runpath, num_retry=1)
            if not loaded and not timed_out:
                # If this idx timed out, returning here would prevent
                # non-successful jobs in being marked as failed (timed out). So
                # return only in the case where it did not time out.
                return

            if loaded:
                jobs = loaded.jobs

        if timed_out:
            for job in jobs:
                if job.status != "Success":
                    job.error = "The run is cancelled due to reaching MAX_RUNTIME"
                    job.status = "Failure"
        self.realization_progress[iteration][run_arg.iens] = jobs, status

    @job_queue({})
    def updateDetailedProgress(self):
        if not self._run_context:
            return

        iteration = self._run_context.get_iter()
        if iteration not in self.realization_progress:
            self.realization_progress[iteration] = {}

        try:
            # Run context might be set to None by concurrent threads,
            # which will results in an Attribute Error
            for idx, run_arg in enumerate(self._run_context):
                self.update_progress_for_index(iteration, idx, run_arg)
        except AttributeError as e:
            if self._run_context is None:
                logging.debug(
                    "Ignoring exception in run model (run_context is None): {}".format(
                        str(e)
                    )
                )
            else:
                raise

    def getDetailedProgress(self):
        self.updateDetailedProgress()

        if (
            self._run_context
            and self._run_context.get_iter() in self.realization_progress
        ):
            return self.realization_progress, self._run_context.get_iter()

        elif self._last_run_iteration in self.realization_progress:
            return self.realization_progress, self._last_run_iteration

        else:
            return {}, -1

    def isIndeterminate(self):
        """@rtype: bool"""
        return not self.isFinished() and self._indeterminate

    def checkHaveSufficientRealizations(self, num_successful_realizations):
        if num_successful_realizations == 0:
            raise ErtRunError("Simulation failed! All realizations failed!")
        elif (
            not self.ert()
            .analysisConfig()
            .haveEnoughRealisations(num_successful_realizations, self._ensemble_size)
        ):
            raise ErtRunError(
                "Too many simulations have failed! You can add/adjust MIN_REALIZATIONS to allow failures in your simulations.\n\n"
                "Check ERT log file '%s' or simulation folder for details."
                % ResLog.getFilename()
            )

    def checkMinimumActiveRealizations(self, run_context):
        active_realizations = self.count_active_realizations(run_context)
        if (
            not self.ert()
            .analysisConfig()
            .haveEnoughRealisations(active_realizations, self._ensemble_size)
        ):
            raise ErtRunError(
                "Number of active realizations is less than the specified MIN_REALIZATIONS in the config file"
            )

    def count_active_realizations(self, run_context):
        return sum(run_context.get_mask())

    def run_ensemble_evaluator(self, run_context, ee_config):
        if run_context.get_step():
            self.ert().eclConfig().assert_restart()

        iactive = run_context.get_mask()

        run_context.get_sim_fs().getStateMap().deselectMatching(
            iactive,
            RealizationStateEnum.STATE_LOAD_FAILURE
            | RealizationStateEnum.STATE_PARENT_FAILURE,
        )

        ensemble = create_ensemble_builder_from_legacy(
            run_context,
            self.get_forward_model(),
            self._queue_config,
            self.ert().analysisConfig(),
            self.ert().resConfig(),
        ).build()

        self.ert().initRun(run_context)

        totalOk = EnsembleEvaluator(
            ensemble,
            ee_config,
            run_context.get_iter(),
            ee_id=str(uuid.uuid1()).split("-")[0],
        ).run_and_get_successful_realizations()

        for i in range(len(run_context)):
            if run_context.is_active(i):
                run_arg = run_context[i]
                if (
                    run_arg.run_status == RunStatusType.JOB_LOAD_FAILURE
                    or run_arg.run_status == RunStatusType.JOB_RUN_FAILURE
                ):
                    run_context.deactivate_realization(i)

        run_context.get_sim_fs().fsync()
        return totalOk

    def get_forward_model(self):
        return self.ert().resConfig().model_config.getForwardModel()

    def get_run_context(self) -> ErtRunContext:
        return self._run_context

    @feature_enabled("new-storage")
    def _post_ensemble_data(self, update_id: Optional[str] = None) -> str:
        self.setPhaseName("Uploading data...")
        ensemble_id = post_ensemble_data(
            ert=ERT.enkf_facade, ensemble_size=self._ensemble_size, update_id=update_id
        )
        self.setPhaseName("Uploading done")
        return ensemble_id

    @feature_enabled("new-storage")
    def _post_ensemble_results(self, ensemble_id: str) -> None:
        self.setPhaseName("Uploading results...")
        post_ensemble_results(ert=ERT.enkf_facade, ensemble_id=ensemble_id)
        self.setPhaseName("Uploading done")

    @feature_enabled("new-storage")
    def _post_update_data(self, parent_ensemble_id: str, algorithm: str) -> str:
        self.setPhaseName("Uploading update...")
        update_id = post_update_data(
            ert=ERT.enkf_facade,
            parent_ensemble_id=parent_ensemble_id,
            algorithm=algorithm,
        )
        self.setPhaseName("Uploading done")
        return update_id
