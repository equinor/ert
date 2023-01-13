from enum import Enum


class RealizationStateEnum(Enum):
    STATE_UNDEFINED = 1
    STATE_INITIALIZED = 2
    STATE_HAS_DATA = 4
    STATE_LOAD_FAILURE = 8
    STATE_PARENT_FAILURE = 16
