from enum import Enum


class ThreadStatus(Enum):
    READY = 1
    RUNNING = 2
    FAILED = 3
    DONE = 4
    STOPPING = 5
