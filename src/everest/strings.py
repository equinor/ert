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
SIMULATION = "simulation"
SIMULATOR_START = "start"
SIMULATOR_UPDATE = "update"
SIMULATOR_END = "end"
SIM_PROGRESS_ID = "simulation_progress"
STORAGE_DIR = "simulation_results"


class EverEndpoints(StrEnum):
    stop = "stop"
    start_experiment = "start_experiment"
    config_path = "config_path"
    start_time = "start_time_unix"
    runs = "runs"
    status = "status"
    events = "events"
    check_runpath = "check_runpath"
    delete_runpaths = "delete_runpaths"
    rerun_failed = "rerun_failed"
    has_failed_realizations = "has_failed_realizations"
