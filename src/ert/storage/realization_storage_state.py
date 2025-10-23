from enum import Enum
from typing import Literal


class RealizationStorageState(Enum):
    UNDEFINED = 1
    PARAMETERS_LOADED = 2
    RESPONSES_LOADED = 4
    FAILURE_IN_CURRENT = 8
    FAILURE_IN_PARENT = 16


RealizationStorageStateNames = Literal[
    "UNDEFINED",
    "PARAMETERS_LOADED",
    "RESPONSES_LOADED",
    "FAILURE_IN_CURRENT",
    "FAILURE_IN_PARENT",
]
