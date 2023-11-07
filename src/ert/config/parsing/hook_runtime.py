import sys

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


class HookRuntime(StrEnum):
    PRE_SIMULATION = "PRE_SIMULATION"
    POST_SIMULATION = "POST_SIMULATION"
    PRE_UPDATE = "PRE_UPDATE"
    POST_UPDATE = "POST_UPDATE"
    PRE_FIRST_UPDATE = "PRE_FIRST_UPDATE"
