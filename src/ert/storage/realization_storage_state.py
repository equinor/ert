from enum import Enum


class RealizationStorageState(Enum):
    UNDEFINED = 1
    PARAMETERS_LOADED = 2
    RESPONSES_LOADED = 4
    FAILURE_IN_CURRENT = 8
    FAILURE_IN_PARENT = 16
