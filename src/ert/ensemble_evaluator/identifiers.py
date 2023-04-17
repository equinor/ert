ACTIVE = "active"
CURRENT_MEMORY_USAGE = "current_memory_usage"
DATA = "data"
END_TIME = "end_time"
ERROR = "error"
ERROR_MSG = "error_msg"
ERROR_FILE = "error_file"
INDEX = "index"
JOBS = "jobs"
MAX_MEMORY_USAGE = "max_memory_usage"
METADATA = "metadata"
NAME = "name"
REALS = "reals"
START_TIME = "start_time"
STATUS = "status"
STDERR = "stderr"
STDOUT = "stdout"
STEPS = "steps"

EVTYPE_FM_STEP_FAILURE = "com.equinor.ert.forward_model_step.failure"
EVTYPE_FM_STEP_PENDING = "com.equinor.ert.forward_model_step.pending"
EVTYPE_FM_STEP_RUNNING = "com.equinor.ert.forward_model_step.running"
EVTYPE_FM_STEP_SUCCESS = "com.equinor.ert.forward_model_step.success"
EVTYPE_FM_STEP_UNKNOWN = "com.equinor.ert.forward_model_step.unknown"
EVTYPE_FM_STEP_WAITING = "com.equinor.ert.forward_model_step.waiting"
EVTYPE_FM_STEP_TIMEOUT = "com.equinor.ert.forward_model_step.timeout"

EVTYPE_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
EVTYPE_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
EVTYPE_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
EVTYPE_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"


EVGROUP_FM_STEP = {
    EVTYPE_FM_STEP_FAILURE,
    EVTYPE_FM_STEP_PENDING,
    EVTYPE_FM_STEP_RUNNING,
    EVTYPE_FM_STEP_SUCCESS,
    EVTYPE_FM_STEP_UNKNOWN,
    EVTYPE_FM_STEP_WAITING,
    EVTYPE_FM_STEP_TIMEOUT,
}

EVGROUP_FM_JOB = {
    EVTYPE_FM_JOB_START,
    EVTYPE_FM_JOB_RUNNING,
    EVTYPE_FM_JOB_SUCCESS,
    EVTYPE_FM_JOB_FAILURE,
}

EVGROUP_FM_ALL = EVGROUP_FM_STEP | EVGROUP_FM_JOB

EVTYPE_EE_SNAPSHOT = "com.equinor.ert.ee.snapshot"
EVTYPE_EE_SNAPSHOT_UPDATE = "com.equinor.ert.ee.snapshot_update"
EVTYPE_EE_TERMINATED = "com.equinor.ert.ee.terminated"
EVTYPE_EE_USER_CANCEL = "com.equinor.ert.ee.user_cancel"
EVTYPE_EE_USER_DONE = "com.equinor.ert.ee.user_done"

EVTYPE_ENSEMBLE_STARTED = "com.equinor.ert.ensemble.started"
EVTYPE_ENSEMBLE_STOPPED = "com.equinor.ert.ensemble.stopped"
EVTYPE_ENSEMBLE_CANCELLED = "com.equinor.ert.ensemble.cancelled"
EVTYPE_ENSEMBLE_FAILED = "com.equinor.ert.ensemble.failed"

EVGROUP_ENSEMBLE = {
    EVTYPE_ENSEMBLE_STARTED,
    EVTYPE_ENSEMBLE_STOPPED,
    EVTYPE_ENSEMBLE_CANCELLED,
    EVTYPE_ENSEMBLE_FAILED,
}
