import time

from math import ceil, trunc
from res.job_queue import JobStatusType


class SimulationStateStatus(object):
    COLOR_WAITING = (164, 200, 255)
    COLOR_PENDING = (190,174,212)
    COLOR_RUNNING = (255,255,153)
    COLOR_FAILED  = (255, 200, 200)
    COLOR_UNKNOWN  = (128, 128, 128)
    COLOR_FINISHED   = (127,201,127)
    COLOR_NOT_ACTIVE  = (255, 255, 255)

    def __init__(self, name, state, color):
        self.__name = name
        self.__state = state
        self.__color = color

        self.__count = 0
        self.__total_count = 1

    @property
    def name(self):
        return self.__name

    @property
    def state(self):
        return self.__state

    @property
    def color(self):
        return self.__color

    @property
    def count(self):
        return self.__count

    @count.setter
    def count(self, value):
        self.__count = value

    @property
    def total_count(self):
        return self.__total_count

    @total_count.setter
    def total_count(self, value):
        self.__total_count = value


class SimulationsTracker(object):
    """SimulationsTracker provide means for tracking a simulation."""
    def __init__(self, model=None, update_interval=0.2, emit_interval=5):
        """Creates a SimulationTracker. Use @model if tracking is to be used.
        The provided @model is then then polled each @update_interval. For each
        @emit_interval, an _update_ is yielded."""
        super(SimulationsTracker, self).__init__()

        waiting_flag  = JobStatusType.JOB_QUEUE_NOT_ACTIVE | JobStatusType.JOB_QUEUE_WAITING | JobStatusType.JOB_QUEUE_SUBMITTED
        waiting_state = SimulationStateStatus("Waiting", waiting_flag, SimulationStateStatus.COLOR_WAITING)

        pending_flag  = JobStatusType.JOB_QUEUE_PENDING
        pending_state = SimulationStateStatus("Pending", pending_flag, SimulationStateStatus.COLOR_PENDING)

        running_flag  = JobStatusType.JOB_QUEUE_RUNNING | JobStatusType.JOB_QUEUE_EXIT | JobStatusType.JOB_QUEUE_RUNNING_DONE_CALLBACK | JobStatusType.JOB_QUEUE_RUNNING_EXIT_CALLBACK
        running_state = SimulationStateStatus("Running", running_flag, SimulationStateStatus.COLOR_RUNNING)

        # Failed also includes simulations which have been killed by the MAX_RUNTIME system.
        failed_flag  = JobStatusType.JOB_QUEUE_IS_KILLED | JobStatusType.JOB_QUEUE_DO_KILL
        failed_flag |= JobStatusType.JOB_QUEUE_FAILED    | JobStatusType.JOB_QUEUE_DO_KILL_NODE_FAILURE
        failed_state = SimulationStateStatus("Failed", failed_flag, SimulationStateStatus.COLOR_FAILED)

        done_flag  = JobStatusType.JOB_QUEUE_DONE | JobStatusType.JOB_QUEUE_SUCCESS
        done_state = SimulationStateStatus("Finished", done_flag, SimulationStateStatus.COLOR_FINISHED)

        unknown_flag  = JobStatusType.JOB_QUEUE_UNKNOWN
        unknown_state = SimulationStateStatus("Unknown", unknown_flag, SimulationStateStatus.COLOR_UNKNOWN)

        self.states = [done_state, failed_state, unknown_state, running_state, pending_state, waiting_state]
        self.custom_states = [done_state, failed_state, running_state, unknown_state, pending_state, waiting_state]
        self.__checkForUnusedEnums()

        self.__current_iteration = 0
        self.__total_iterations = 0
        self.__iteration_name = ""
        self.__runtime = 0  # seconds
        self.__queue_size = 0
        self.__indeterminate = False

        self._update_interval = update_interval
        self._emit_interval = emit_interval

        self._model = model

    def getStates(self):
        """ @rtype: list[SimulationStateStatus] """
        return list(self.custom_states)

    def __checkForUnusedEnums(self):
        for enum in JobStatusType.enums():
            # The status check routines can return this status; if e.g. the bjobs call fails,
            # but a job will never get this status.
            if enum == JobStatusType.JOB_QUEUE_STATUS_FAILURE:
                continue
                
                
            used = False
            for state in self.states:
                if enum in state.state:
                    used = True

            if not used:
                raise AssertionError("Enum identifier '%s' not used!" % enum)

    def _update(self):
        self.__current_iteration = self._model.currentPhase()
        self.__total_iterations = self._model.phaseCount()
        self.__queue_size = self._model.getQueueSize()
        self.__iteration_name = self._model.getPhaseName()
        self.__runtime = self._model.getRunningTime()
        self.__indeterminate = self._model.isIndeterminate()

        queue_status = self._model.getQueueStatus()

        for state in self.getStates():
            state.count = 0
            state.total_count = self.__queue_size

            for queue_state in queue_status:
                if queue_state in state.state:
                    state.count += queue_status[queue_state]

    def track(self):
        """Tracks a model and provides _updates_, which currently is the
        instance itself."""
        if self._model is None:
            raise ValueError("no model to track")

        while not self._model.isFinished():
            self._update()

            runtime = trunc(self.__runtime)
            if runtime % self._emit_interval == 0:
                yield self

                # Sleep for a whole second so as to not emit multiple updates
                # within this emit interval.
                time.sleep(ceil(self._update_interval))

            time.sleep(self._update_interval)

        # Simulations are done, do final update and emit.
        self._update()
        yield self

    @property
    def run_failed(self):
        return self._model.hasRunFailed()

    @property
    def failed_message(self):
        return self._model.getFailMessage()

    @property
    def current_iteration(self):
        return self.__current_iteration

    @property
    def total_iterations(self):
        return self.__total_iterations

    @property
    def iteration_name(self):
        return self.__iteration_name

    @property
    def runtime(self):
        return self.__runtime

    @property
    def queue_size(self):
        return self.__queue_size

    @property
    def indeterminate(self):
        return self.____indeterminate

    @staticmethod
    def format_running_time(runtime):
        """ @rtype: str """
        days = 0
        hours = 0
        minutes = 0
        seconds = trunc(runtime)

        if seconds >= 60:
            minutes, seconds = divmod(seconds, 60)

        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)

        if hours >= 24:
            days, hours = divmod(hours, 24)

        if days > 0:
            layout = "Running time: {d} days {h} hours {m} minutes {s} seconds"

        elif hours > 0:
            layout = "Running time: {h} hours {m} minutes {s} seconds"

        elif minutes > 0:
            layout = "Running time: {m} minutes {s} seconds"

        else:
            layout = "Running time: {s} seconds"

        return layout.format(d=days, h=hours, m=minutes, s=seconds)
