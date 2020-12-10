from res.job_queue import JobStatusType

REALIZATION_STATE_WAITING = "Waiting"
REALIZATION_STATE_PENDING = "Pending"
REALIZATION_STATE_RUNNING = "Running"
REALIZATION_STATE_FAILED = "Failed"
REALIZATION_STATE_FINISHED = "Finished"
REALIZATION_STATE_UNKNOWN = "Unknown"

ALL_REALIZATION_STATES = (
    REALIZATION_STATE_WAITING,
    REALIZATION_STATE_PENDING,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_FINISHED,
    REALIZATION_STATE_UNKNOWN,
)


COLOR_WAITING = (164, 200, 255)
COLOR_PENDING = (190, 174, 212)
COLOR_RUNNING = (255, 255, 153)
COLOR_FAILED = (255, 200, 200)
COLOR_UNKNOWN = (128, 128, 128)
COLOR_FINISHED = (127, 201, 127)
COLOR_NOT_ACTIVE = (255, 255, 255)


REAL_STATE_TO_COLOR = {
    REALIZATION_STATE_WAITING: COLOR_WAITING,
    REALIZATION_STATE_PENDING: COLOR_PENDING,
    REALIZATION_STATE_RUNNING: COLOR_RUNNING,
    REALIZATION_STATE_FAILED: COLOR_FAILED,
    REALIZATION_STATE_FINISHED: COLOR_FINISHED,
    REALIZATION_STATE_UNKNOWN: COLOR_UNKNOWN
}

QUEUE_WAITING_FLAG = (
    JobStatusType.JOB_QUEUE_NOT_ACTIVE
    | JobStatusType.JOB_QUEUE_WAITING
    | JobStatusType.JOB_QUEUE_SUBMITTED
)
QUEUE_PENDING_FLAG = JobStatusType.JOB_QUEUE_PENDING
QUEUE_RUNNING_FLAG = (
    JobStatusType.JOB_QUEUE_RUNNING
    | JobStatusType.JOB_QUEUE_EXIT
    | JobStatusType.JOB_QUEUE_RUNNING_DONE_CALLBACK
    | JobStatusType.JOB_QUEUE_RUNNING_EXIT_CALLBACK
)
# Failed also includes simulations which have been killed by the MAX_RUNTIME system.
QUEUE_FAILED_FLAG = JobStatusType.JOB_QUEUE_IS_KILLED | JobStatusType.JOB_QUEUE_DO_KILL
QUEUE_FAILED_FLAG |= (
    JobStatusType.JOB_QUEUE_FAILED | JobStatusType.JOB_QUEUE_DO_KILL_NODE_FAILURE
)
QUEUE_DONE_FLAG = JobStatusType.JOB_QUEUE_DONE | JobStatusType.JOB_QUEUE_SUCCESS
QUEUE_UNKNOWN_FLAG = JobStatusType.JOB_QUEUE_UNKNOWN

QUEUE_FLAG_TO_REAL_STATE = {
    QUEUE_WAITING_FLAG: REALIZATION_STATE_WAITING,
    QUEUE_PENDING_FLAG: REALIZATION_STATE_PENDING,
    QUEUE_RUNNING_FLAG: REALIZATION_STATE_RUNNING,
    QUEUE_FAILED_FLAG: REALIZATION_STATE_FAILED,
    QUEUE_DONE_FLAG: REALIZATION_STATE_FINISHED,
    QUEUE_UNKNOWN_FLAG: REALIZATION_STATE_UNKNOWN,
}


def queue_status_to_real_state(queue_status):
    for flag in QUEUE_FLAG_TO_REAL_STATE.keys():
        if queue_status in flag:
            return QUEUE_FLAG_TO_REAL_STATE[flag]
    raise ValueError(f"could not map queue status {queue_status} to state")

# class SimulationStateStatus(object):
#     def __init__(self, name, state, color):
#         self.__name = name
#         self.__state = state
#         self.__color = color

#         self.__count = 0
#         self.__total_count = 1

#     @property
#     def name(self):
#         return self.__name

#     @property
#     def state(self):
#         return self.__state

#     @property
#     def color(self):
#         return self.__color

#     @property
#     def count(self):
#         return self.__count

#     @count.setter
#     def count(self, value):
#         self.__count = value

#     @property
#     def total_count(self):
#         return self.__total_count

#     @total_count.setter
#     def total_count(self, value):
#         self.__total_count = value


# def create_states():
#     waiting_state = SimulationStateStatus(
#         REALIZATION_STATE_WAITING,
#         QUEUE_WAITING_FLAG,
#         COLOR_WAITING,
#     )

#     pending_state = SimulationStateStatus(
#         REALIZATION_STATE_PENDING,
#         QUEUE_PENDING_FLAG,
#         COLOR_PENDING,
#     )

#     running_state = SimulationStateStatus(
#         REALIZATION_STATE_RUNNING,
#         QUEUE_RUNNING_FLAG,
#         COLOR_RUNNING,
#     )

#     failed_state = SimulationStateStatus(
#         REALIZATION_STATE_FAILED,
#         QUEUE_FAILED_FLAG,
#         COLOR_FAILED,
#     )

#     done_state = SimulationStateStatus(
#         REALIZATION_STATE_FINISHED,
#         QUEUE_DONE_FLAG,
#         COLOR_FINISHED,
#     )

#     unknown_state = SimulationStateStatus(
#         REALIZATION_STATE_UNKNOWN,
#         QUEUE_UNKNOWN_FLAG,
#         COLOR_UNKNOWN,
#     )
#     return [
#         done_state,
#         failed_state,
#         running_state,
#         unknown_state,
#         pending_state,
#         waiting_state,
#     ]
