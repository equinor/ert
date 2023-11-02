import sys

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


class WorkflowJobKeys(StrEnum):
    MIN_ARG = "MIN_ARG"
    MAX_ARG = "MAX_ARG"
    ARG_TYPE = "ARG_TYPE"
    ARGLIST = "ARGLIST"
    EXECUTABLE = "EXECUTABLE"
    SCRIPT = "SCRIPT"
    INTERNAL = "INTERNAL"
    STOP_ON_FAIL = "STOP_ON_FAIL"


class ConfigArgAtIndex(StrEnum):
    pass
