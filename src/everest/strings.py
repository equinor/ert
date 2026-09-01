from enum import StrEnum, auto

DEFAULT_OUTPUT_DIR = "everest_output"
DEFAULT_LOGGING_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"

EVEREST = "everest"
NAME = "name"

OPTIMIZATION_OUTPUT_DIR = "optimization_output"
OPTIMIZATION_LOG_DIR = "logs"
OPT_PROGRESS_ID = "optimization_progress"
OPT_FAILURE_REALIZATIONS = (
    "Optimization failed: not enough successful realizations to proceed."
)
OPT_FAILURE_ALL_REALIZATIONS = "Optimization failed: all realizations failed."

SESSION_DIR = ".session"
SIM_PROGRESS_ID = "simulation_progress"
STORAGE_DIR = "simulation_results"


class EverEndpoints(StrEnum):
    STOP = auto()
    START_EXPERIMENT = auto()
    CONFIG_PATH = auto()
    START_TIME = auto()
    EXPERIMENTS = auto()
    STATUS = auto()
    EVENTS = auto()
    RUNPATH = auto()
