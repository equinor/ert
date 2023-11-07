import sys

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


class AnalysisMode(StrEnum):
    ITERATED_ENSEMBLE_SMOOTHER = "IES_ENKF"
    ENSEMBLE_SMOOTHER = "STD_ENKF"
