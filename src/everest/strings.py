from enum import StrEnum

CERTIFICATE_DIR = "cert"

DETACHED_NODE_DIR = "detached_node_output"
DEFAULT_OUTPUT_DIR = "everest_output"
DEFAULT_LOGGING_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"

EVEREST = "everest"
EVERSERVER = "everserver"

HOSTFILE_NAME = "storage_server.json"

NAME = "name"

OPTIMIZATION_OUTPUT_DIR = "optimization_output"
OPTIMIZATION_LOG_DIR = "logs"
OPT_PROGRESS_ID = "optimization_progress"
OPT_FAILURE_REALIZATIONS = (
    "Optimization failed: not enough successful realizations to proceed."
)
OPT_FAILURE_ALL_REALIZATIONS = "Optimization failed: all realizations failed."

SESSION_DIR = ".session"
SERVER_STATUS = "status"
SIMULATION = "simulation"
SIMULATION_DIR = "simulation_<IENS>"
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
