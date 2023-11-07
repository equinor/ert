import sys

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


class HistorySource(StrEnum):
    REFCASE_SIMULATED = "REFCASE_SIMULATED"
    REFCASE_HISTORY = "REFCASE_HISTORY"
