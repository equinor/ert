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

EVTYPE_REALIZATION_FAILURE = "com.equinor.ert.realization.failure"
EVTYPE_REALIZATION_PENDING = "com.equinor.ert.realization.pending"
EVTYPE_REALIZATION_RUNNING = "com.equinor.ert.realization.running"
EVTYPE_REALIZATION_SUCCESS = "com.equinor.ert.realization.success"
EVTYPE_REALIZATION_UNKNOWN = "com.equinor.ert.realization.unknown"
EVTYPE_REALIZATION_WAITING = "com.equinor.ert.realization.waiting"
EVTYPE_REALIZATION_TIMEOUT = "com.equinor.ert.realization.timeout"

EVTYPE_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
EVTYPE_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
EVTYPE_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
EVTYPE_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"


EVGROUP_REALIZATION = {
    EVTYPE_REALIZATION_FAILURE,
    EVTYPE_REALIZATION_PENDING,
    EVTYPE_REALIZATION_RUNNING,
    EVTYPE_REALIZATION_SUCCESS,
    EVTYPE_REALIZATION_UNKNOWN,
    EVTYPE_REALIZATION_WAITING,
    EVTYPE_REALIZATION_TIMEOUT,
}

EVGROUP_FM_JOB = {
    EVTYPE_FM_JOB_START,
    EVTYPE_FM_JOB_RUNNING,
    EVTYPE_FM_JOB_SUCCESS,
    EVTYPE_FM_JOB_FAILURE,
}

EVGROUP_FM_ALL = EVGROUP_REALIZATION | EVGROUP_FM_JOB

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
