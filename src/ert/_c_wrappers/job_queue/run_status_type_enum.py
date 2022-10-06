from cwrap import BaseCEnum


class RunStatusType(BaseCEnum):
    TYPE_NAME = "run_status_type_enum"

    JOB_LOAD_FAILURE = None
    JOB_RUN_FAILURE = None

    @classmethod
    def from_string(cls, name):
        pass


RunStatusType.addEnum("JOB_RUN_FAILURE", 2)
RunStatusType.addEnum("JOB_LOAD_FAILURE", 3)
