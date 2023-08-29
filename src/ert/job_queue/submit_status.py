from cwrap import BaseCEnum


class SubmitStatus(BaseCEnum):  # type: ignore
    TYPE_NAME = "job_submit_status_type_enum"
    OK = None
    JOB_FAIL = None
    DRIVER_FAIL = None
    QUEUE_CLOSED = None

    @classmethod
    def from_string(cls, name: str) -> "SubmitStatus":
        return super().from_string(name)


SubmitStatus.addEnum("OK", 0)
SubmitStatus.addEnum("JOB_FAIL", 1)
SubmitStatus.addEnum("DRIVER_FAIL", 2)
SubmitStatus.addEnum("QUEUE_CLOSED", 3)
