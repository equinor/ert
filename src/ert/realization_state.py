from enum import Enum


class RealizationStorageState(Enum):
    UNDEFINED = 1
    INITIALIZED = 2
    HAS_DATA = 4
    LOAD_FAILURE = 8
    PARENT_FAILURE = 16
