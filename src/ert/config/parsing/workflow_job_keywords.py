from ert.enum_shim import StrEnum


class WorkflowJobKeys(StrEnum):
    MIN_ARG = "MIN_ARG"
    MAX_ARG = "MAX_ARG"
    ARG_TYPE = "ARG_TYPE"
    ARGLIST = "ARGLIST"
    EXECUTABLE = "EXECUTABLE"
    SCRIPT = "SCRIPT"
    INTERNAL = "INTERNAL"
    STOP_ON_FAIL = "STOP_ON_FAIL"
