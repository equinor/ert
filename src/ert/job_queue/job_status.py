from cwrap import BaseCEnum


class JobStatus(BaseCEnum):  # type: ignore
    TYPE_NAME = "job_status_type_enum"
    # This value is used in external query routines - for jobs which are
    # (currently) not active.
    NOT_ACTIVE = None
    WAITING = None  # A node which is waiting in the internal queue.
    # Internal status: It has has been submitted - the next status update will
    # (should) place it as pending or running.
    SUBMITTED = None
    # A node which is pending - a status returned by the external system. I.e LSF
    PENDING = None
    RUNNING = None  # The job is running
    # The job is done - but we have not yet checked if the target file is
    # produced
    DONE = None
    # The job has exited - check attempts to determine if we retry or go to
    # complete_fail
    EXIT = None
    # The job has been killed, following a  DO_KILL - can restart.
    IS_KILLED = None
    # The the job should be killed, either due to user request, or automated
    # measures - the job can NOT be restarted..
    DO_KILL = None
    SUCCESS = None
    STATUS_FAILURE = None
    FAILED = None
    DO_KILL_NODE_FAILURE = None
    UNKNOWN = None

    @classmethod
    def from_string(cls, name: str) -> "JobStatus":
        return super().from_string(name)


JobStatus.addEnum("NOT_ACTIVE", 1)
JobStatus.addEnum("WAITING", 2)
JobStatus.addEnum("SUBMITTED", 4)
JobStatus.addEnum("PENDING", 8)
JobStatus.addEnum("RUNNING", 16)
JobStatus.addEnum("DONE", 32)
JobStatus.addEnum("EXIT", 64)
JobStatus.addEnum("IS_KILLED", 128)
JobStatus.addEnum("DO_KILL", 256)
JobStatus.addEnum("SUCCESS", 512)
JobStatus.addEnum("STATUS_FAILURE", 1024)
JobStatus.addEnum("FAILED", 2048)
JobStatus.addEnum("DO_KILL_NODE_FAILURE", 4096)
JobStatus.addEnum("UNKNOWN", 8192)
