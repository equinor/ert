from enum import Enum


class RealizationStorageState(Enum):
    UNDEFINED = 1
    PARAMETERS_LOADED = 2
    RESPONSES_LOADED = 4
    LOAD_FAILURE = 8
    PARENT_FAILURE = 16
