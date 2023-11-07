from enum import Enum


class ThreadStatus(Enum):
    READY = 1  # JobStatus.WAITING, SUBMITTED, PENDING ?
    RUNNING = 2  # JobStatus.RUNNING
    FAILED = 3  # IS_KILLED, FAILED
    DONE = 4  # DONE, SUCCESS
    STOPPING = 5  # DO_KILL

    # JobStatus.NOT_ACTIVE is not relevant here.
