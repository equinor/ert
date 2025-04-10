from enum import StrEnum

CERTIFICATE_DIR = "cert"

DATE_FORMAT = "%Y-%m-%d"
DETACHED_NODE_DIR = "detached_node_output"
DEFAULT_OUTPUT_DIR = "everest_output"
DEFAULT_LOGGING_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"

EVEREST_SERVER_CONFIG = "everserver_config"
EVEREST = "everest"
EVERSERVER = "everserver"

HOSTFILE_NAME = "hostfile"

NAME = "name"

OPTIMIZATION_OUTPUT_DIR = "optimization_output"
OPTIMIZATION_LOG_DIR = "logs"
OPT_PROGRESS_ID = "optimization_progress"
OPT_FAILURE_REALIZATIONS = (
    "Optimization failed: not enough successful realizations to proceed."
)

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
    simulation_dir = "simulation_dir"
    start_time = "start_time_unix"
