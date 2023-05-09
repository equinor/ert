from enum import Enum


class HookRuntime(Enum):
    PRE_SIMULATION = 0
    POST_SIMULATION = 1
    PRE_UPDATE = 2
    POST_UPDATE = 3
    PRE_FIRST_UPDATE = 4
