from typing import Final

from ert.event_type_constants import (
    EVTYPE_ENSEMBLE_CANCELLED,
    EVTYPE_ENSEMBLE_FAILED,
    EVTYPE_ENSEMBLE_STARTED,
    EVTYPE_ENSEMBLE_STOPPED,
    EVTYPE_REALIZATION_FAILURE,
    EVTYPE_REALIZATION_PENDING,
    EVTYPE_REALIZATION_RUNNING,
    EVTYPE_REALIZATION_SUCCESS,
    EVTYPE_REALIZATION_TIMEOUT,
    EVTYPE_REALIZATION_UNKNOWN,
    EVTYPE_REALIZATION_WAITING,
)

ACTIVE: Final = "active"
CURRENT_MEMORY_USAGE: Final = "current_memory_usage"
DATA: Final = "data"
END_TIME: Final = "end_time"
ERROR: Final = "error"
ERROR_MSG: Final = "error_msg"
ERROR_FILE: Final = "error_file"
INDEX: Final = "index"
JOBS: Final = "jobs"
MAX_MEMORY_USAGE: Final = "max_memory_usage"
METADATA: Final = "metadata"
NAME: Final = "name"
REALS: Final = "reals"
START_TIME: Final = "start_time"
STATUS: Final = "status"
STDERR: Final = "stderr"
STDOUT: Final = "stdout"
STEPS: Final = "steps"

EVTYPE_FORWARD_MODEL_START: Final = "com.equinor.ert.forward_model_job.start"
EVTYPE_FORWARD_MODEL_RUNNING: Final = "com.equinor.ert.forward_model_job.running"
EVTYPE_FORWARD_MODEL_SUCCESS: Final = "com.equinor.ert.forward_model_job.success"
EVTYPE_FORWARD_MODEL_FAILURE: Final = "com.equinor.ert.forward_model_job.failure"
EVTYPE_FORWARD_MODEL_CHECKSUM: Final = "com.equinor.ert.forward_model_job.checksum"


EVGROUP_REALIZATION: Final = {
    EVTYPE_REALIZATION_FAILURE,
    EVTYPE_REALIZATION_PENDING,
    EVTYPE_REALIZATION_RUNNING,
    EVTYPE_REALIZATION_SUCCESS,
    EVTYPE_REALIZATION_UNKNOWN,
    EVTYPE_REALIZATION_WAITING,
    EVTYPE_REALIZATION_TIMEOUT,
}

EVGROUP_FORWARD_MODEL: Final = {
    EVTYPE_FORWARD_MODEL_START,
    EVTYPE_FORWARD_MODEL_RUNNING,
    EVTYPE_FORWARD_MODEL_SUCCESS,
    EVTYPE_FORWARD_MODEL_FAILURE,
}

EVGROUP_FM_ALL = EVGROUP_REALIZATION | EVGROUP_FORWARD_MODEL

EVTYPE_EE_SNAPSHOT: Final = "com.equinor.ert.ee.snapshot"
EVTYPE_EE_SNAPSHOT_UPDATE: Final = "com.equinor.ert.ee.snapshot_update"
EVTYPE_EE_TERMINATED: Final = "com.equinor.ert.ee.terminated"
EVTYPE_EE_USER_CANCEL: Final = "com.equinor.ert.ee.user_cancel"
EVTYPE_EE_USER_DONE: Final = "com.equinor.ert.ee.user_done"


EVGROUP_ENSEMBLE: Final = {
    EVTYPE_ENSEMBLE_STARTED,
    EVTYPE_ENSEMBLE_STOPPED,
    EVTYPE_ENSEMBLE_CANCELLED,
    EVTYPE_ENSEMBLE_FAILED,
}
