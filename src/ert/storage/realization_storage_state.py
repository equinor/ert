from enum import StrEnum, auto


class RealizationStorageState(StrEnum):
    UNDEFINED = auto()
    PARAMETERS_LOADED = auto()
    RESPONSES_LOADED = auto()
    FAILURE_IN_CURRENT = auto()
    FAILURE_IN_PARENT = auto()
