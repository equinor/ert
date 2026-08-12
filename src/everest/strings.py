from enum import StrEnum

DEFAULT_OUTPUT_DIR = "everest_output"
DEFAULT_LOGGING_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"

EVEREST = "everest"
EVERSERVER = "everserver"
EXPERIMENT_SERVER = "experiment_server"

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
    stop = "stop"
    start_experiment = "start_experiment"
    config_path = "config_path"
    start_time = "start_time_unix"
    experiments = "experiments"
    status = "status"
    events = "events"
    runpath = "runpath"
