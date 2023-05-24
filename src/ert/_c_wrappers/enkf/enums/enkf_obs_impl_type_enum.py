from enum import Enum, auto


class EnkfObservationImplementationType(Enum):
    GEN_OBS = auto()
    SUMMARY_OBS = auto()
