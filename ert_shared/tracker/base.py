from ert_shared.tracker.evaluator import EvaluatorTracker
import time

from res.job_queue import JobStatusType

from ert_shared.tracker.events import DetailedEvent, EndEvent, GeneralEvent
from ert_shared.tracker.state import SimulationStateStatus
from ert_shared.tracker.utils import calculate_progress


class BaseTracker:
    """BaseTracker provides the basis for doing tracking."""

    def __init__(self, model, ee_monitor_connection_details):
        """Initialize the tracker for a @model. A model can be any
        BaseRunModel-derived class."""
        self._model = model

        self._states = []
        self._custom_states = []
        self._bootstrap_states()
        BaseTracker.__checkForUnusedEnums(self._states)

        # keep track of phases in the model using a mapping of phase index
        # to data is not accessible in the model, but should be.
        # TODO: rewrite the phases API in BaseRunModel so that this can go away
        #       see https://github.com/equinor/ert/issues/556
        self._phase_states = {}

        self._evaluator_tracker = None
        if ee_monitor_connection_details:
            self._evaluator_tracker = EvaluatorTracker(
                model, ee_monitor_connection_details, self._custom_states
            )

    def _bootstrap_states(self):
        waiting_flag = (
            JobStatusType.JOB_QUEUE_NOT_ACTIVE
            | JobStatusType.JOB_QUEUE_WAITING
            | JobStatusType.JOB_QUEUE_SUBMITTED
        )
        waiting_state = SimulationStateStatus(
            "Waiting", waiting_flag, SimulationStateStatus.COLOR_WAITING
        )

        pending_flag = JobStatusType.JOB_QUEUE_PENDING
        pending_state = SimulationStateStatus(
            "Pending", pending_flag, SimulationStateStatus.COLOR_PENDING
        )

        running_flag = (
            JobStatusType.JOB_QUEUE_RUNNING
            | JobStatusType.JOB_QUEUE_EXIT
            | JobStatusType.JOB_QUEUE_RUNNING_DONE_CALLBACK
            | JobStatusType.JOB_QUEUE_RUNNING_EXIT_CALLBACK
        )
        running_state = SimulationStateStatus(
            "Running", running_flag, SimulationStateStatus.COLOR_RUNNING
        )

        # Failed also includes simulations which have been killed by the MAX_RUNTIME system.
        failed_flag = (
            JobStatusType.JOB_QUEUE_IS_KILLED | JobStatusType.JOB_QUEUE_DO_KILL
        )
        failed_flag |= (
            JobStatusType.JOB_QUEUE_FAILED
            | JobStatusType.JOB_QUEUE_DO_KILL_NODE_FAILURE
        )
        failed_state = SimulationStateStatus(
            "Failed", failed_flag, SimulationStateStatus.COLOR_FAILED
        )

        done_flag = JobStatusType.JOB_QUEUE_DONE | JobStatusType.JOB_QUEUE_SUCCESS
        done_state = SimulationStateStatus(
            "Finished", done_flag, SimulationStateStatus.COLOR_FINISHED
        )

        unknown_flag = JobStatusType.JOB_QUEUE_UNKNOWN
        unknown_state = SimulationStateStatus(
            "Unknown", unknown_flag, SimulationStateStatus.COLOR_UNKNOWN
        )

        self._states = [
            done_state,
            failed_state,
            unknown_state,
            running_state,
            pending_state,
            waiting_state,
        ]
        self._custom_states = [
            done_state,
            failed_state,
            running_state,
            unknown_state,
            pending_state,
            waiting_state,
        ]

    def _update_phase_map(self):
        phase = self._model.currentPhase()
        if phase not in self._phase_states:
            # False indicates that we will not determine (yet) whether or not
            # it has had activity (i.e. queue has run).
            self._phase_states[phase] = False

        if self._model.isQueueRunning():
            # This phase has job queue activity.
            self._phase_states[phase] = True

    def _general_event(self):
        if self._evaluator_tracker is None:
            return self._general_event_from_model()
        else:
            return self._evaluator_tracker.general_event()

    def _general_event_from_model(self):
        self._update_phase_map()

        phase_name = self._model.getPhaseName()
        phase = self._model.currentPhase()
        phase_count = self._model.phaseCount()
        queue_status = self._model.getQueueStatus()

        done_count = 0
        for state in self.get_states():
            state.count = 0
            state.total_count = self._model.getQueueSize()

            for queue_state in queue_status:
                if queue_state in state.state:
                    state.count += queue_status[queue_state]

            if state.name == "Finished":
                done_count = state.count

        progress = calculate_progress(
            phase,
            phase_count,
            self._model.isFinished(),
            self._model.isQueueRunning(),
            self._model.getQueueSize(),
            self._phase_states[phase],
            done_count,
        )

        return GeneralEvent(
            phase_name,
            phase,
            phase_count,
            progress,
            self._model.isIndeterminate(),
            self.get_states(),
            self._model.get_runtime(),
        )

    def _detailed_event(self):
        if self._evaluator_tracker is None:
            return self._detailed_event_from_model()
        else:
            return self._evaluator_tracker.detailed_event()

    def _detailed_event_from_model(self):
        return DetailedEvent(*self._model.getDetailedProgress())

    def _end_event(self):
        return EndEvent(self._model.hasRunFailed(), self._model.getFailMessage())

    def is_finished(self):
        if self._evaluator_tracker is None:
            return self._model.isFinished()
        else:
            return self._evaluator_tracker.is_finished()

    def request_termination(self):
        if self._evaluator_tracker is None:
            return self._model.killAllSimulations()
        else:
            return self._evaluator_tracker.request_termination()

    @staticmethod
    def __checkForUnusedEnums(states):
        for enum in JobStatusType.enums():
            # The status check routines can return this status; if e.g. the bjobs call fails,
            # but a job will never get this status.
            if enum == JobStatusType.JOB_QUEUE_STATUS_FAILURE:
                continue

            used = False
            for state in states:
                if enum in state.state:
                    used = True

            if not used:
                raise AssertionError("Enum identifier '%s' not used!" % enum)

    def get_states(self):
        """ @rtype: list[SimulationStateStatus] """
        return list(self._custom_states)

    def reset(self):
        self._phase_states = {}
