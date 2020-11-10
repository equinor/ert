FM_JOB_ID = "id"
FM_JOB_TYPE = "type"
FM_JOB_INPUTS = "inputs"

TERMINATE = "terminate"
PAUSE = "pause"
ACTION = "action"
TERMINATED = "terminated"
DONE = "done"
STATUS = "status"
FORWARD_MODELS = "forward_models"
REALIZATIONS = "realizations"
EVENT_INDEX = "event_index"
CREATED = "created"

REALIZATION_STATUS = "status"
REALIZATION_FORWARD_MODELS = "forward_models"

EVTYPE_FM_STAGE_WAITING = "com.equinor.ert.forward_model_stage.waiting"
EVTYPE_FM_STAGE_PENDING = "com.equinor.ert.forward_model_stage.pending"
EVTYPE_FM_STAGE_RUNNING = "com.equinor.ert.forward_model_stage.running"
EVTYPE_FM_STAGE_FAILURE = "com.equinor.ert.forward_model_stage.failure"
EVTYPE_FM_STAGE_SUCCESS = "com.equinor.ert.forward_model_stage.success"
EVTYPE_FM_STAGE_UNKNOWN = "com.equinor.ert.forward_model_stage.unknown"

EVTYPE_FM_STEP_START = "com.equinor.ert.forward_model_step.start"
EVTYPE_FM_STEP_FAILURE = "com.equinor.ert.forward_model_step.failure"
EVTYPE_FM_STEP_SUCCESS = "com.equinor.ert.forward_model_step.success"

EVTYPE_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
EVTYPE_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
EVTYPE_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
EVTYPE_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"

EVGROUP_FM_STAGE = {
    EVTYPE_FM_STAGE_WAITING,
    EVTYPE_FM_STAGE_PENDING,
    EVTYPE_FM_STAGE_RUNNING,
    EVTYPE_FM_STAGE_FAILURE,
    EVTYPE_FM_STAGE_SUCCESS,
    EVTYPE_FM_STAGE_UNKNOWN,
}

EVGROUP_FM_STEP = {
    EVTYPE_FM_STEP_START,
    EVTYPE_FM_STEP_FAILURE,
    EVTYPE_FM_STEP_SUCCESS,
}

EVGROUP_FM_JOB = {
    EVTYPE_FM_JOB_START,
    EVTYPE_FM_JOB_RUNNING,
    EVTYPE_FM_JOB_SUCCESS,
    EVTYPE_FM_JOB_FAILURE,
}

EVGROUP_FM_ALL = EVGROUP_FM_STAGE | EVGROUP_FM_STEP | EVGROUP_FM_JOB

EVTYPE_EE_SNAPSHOT = "com.equinor.ert.ee.snapshot"
EVTYPE_EE_SNAPSHOT_UPDATE = "com.equinor.ert.ee.snapshot_update"
EVTYPE_EE_TERMINATED = "com.equinor.ert.ee.terminated"
EVTYPE_EE_USER_CANCEL = "com.equinor.ert.ee.user_cancel"
EVTYPE_EE_USER_DONE = "com.equinor.ert.ee.user_done"

FM_JOB_ATTR_NAME = "name"
FM_JOB_ATTR_EXECUTABLE = "executable"
FM_JOB_ATTR_TARGET_FILE = "target_file"
FM_JOB_ATTR_ERROR_FILE = "error_file"
FM_JOB_ATTR_START_FILE = "start_file"
FM_JOB_ATTR_STDOUT = "stdout"
FM_JOB_ATTR_STDERR = "stderr"
FM_JOB_ATTR_STDIN = "stdin"
FM_JOB_ATTR_ARGLIST = "argList"
FM_JOB_ATTR_ENVIRONMENT = "environment"
FM_JOB_ATTR_EXEC_ENV = "exec_env"
FM_JOB_ATTR_LICENSE_PATH = "license_path"
FM_JOB_ATTR_MAX_RUNNING_MINUTES = "max_running_minutes"
FM_JOB_ATTR_MAX_RUNNING = "max_running"
FM_JOB_ATTR_MIN_ARG = "min_arg"
FM_JOB_ATTR_ARG_TYPES = "arg_types"
FM_JOB_ATTR_MAX_ARG = "max_arg"
FM_JOB_ATTR_STATUS = "status"
FM_JOB_ATTR_CURRENT_MEMORY_USAGE = "current_memory_usage"


EVTYPE_ENSEMBLE_STARTED = "com.equinor.ert.ensemble.started"
EVTYPE_ENSEMBLE_STOPPED = "com.equinor.ert.ensemble.stopped"
EVTYPE_ENSEMBLE_CANCELLED = "com.equinor.ert.ensemble.cancelled"

EVGROUP_ENSEMBLE = {
    EVTYPE_ENSEMBLE_STARTED,
    EVTYPE_ENSEMBLE_STOPPED,
    EVTYPE_ENSEMBLE_CANCELLED,
}

STATUS_QUEUE_STATE = {
    "Waiting": "JOB_QUEUE_WAITING",
    "Finished": "JOB_QUEUE_SUCCESS",
    "Failed": "JOB_QUEUE_FAILED",
    "Pending": "JOB_QUEUE_RUNNING",
    "Running": "JOB_QUEUE_RUNNING",
    "Unknown": "JOB_QUEUE_UNKNOWN",
}
