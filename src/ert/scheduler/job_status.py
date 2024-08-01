from enum import Enum, auto


class JobStatus(Enum):
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
