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


class ConfigArgAtIndex(StrEnum):
    pass


class RunModes(ConfigArgAtIndex):
    RUN_MODE_PRE_SIMULATION_NAME = "PRE_SIMULATION"
    RUN_MODE_POST_SIMULATION_NAME = "POST_SIMULATION"
    RUN_MODE_PRE_UPDATE_NAME = "PRE_UPDATE"
    RUN_MODE_POST_UPDATE_NAME = "POST_UPDATE"
    RUN_MODE_PRE_FIRST_UPDATE_NAME = "PRE_FIRST_UPDATE"


class QueueOptions(ConfigArgAtIndex):
    LSF = "LSF"
    LOCAL = "LOCAL"
    TORQUE = "TORQUE"
    SLURM = "SLURM"
