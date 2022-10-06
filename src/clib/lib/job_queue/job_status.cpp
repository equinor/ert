#include <ert/job_queue/job_status.hpp>

const char *job_status_get_name(job_status_type status) {
    switch (status) {
    case JOB_QUEUE_NOT_ACTIVE:
        return "JOB_QUEUE_NOT_ACTIVE";
        break;
    case JOB_QUEUE_WAITING:
        return "JOB_QUEUE_WAITING";
        break;
    case JOB_QUEUE_SUBMITTED:
        return "JOB_QUEUE_SUBMITTED";
        break;
    case JOB_QUEUE_PENDING:
        return "JOB_QUEUE_PENDING";
        break;
    case JOB_QUEUE_RUNNING:
        return "JOB_QUEUE_RUNNING";
        break;
    case JOB_QUEUE_DONE:
        return "JOB_QUEUE_DONE";
        break;
    case JOB_QUEUE_EXIT:
        return "JOB_QUEUE_EXIT";
        break;
    case JOB_QUEUE_IS_KILLED:
        return "JOB_QUEUE_IS_KILLED";
        break;
    case JOB_QUEUE_DO_KILL:
        return "JOB_QUEUE_DO_KILL";
        break;
    case JOB_QUEUE_SUCCESS:
        return "JOB_QUEUE_SUCCESS";
        break;
    case JOB_QUEUE_RUNNING_DONE_CALLBACK:
        return "JOB_QUEUE_RUNNING_DONE_CALLBACK";
        break;
    case JOB_QUEUE_RUNNING_EXIT_CALLBACK:
        return "JOB_QUEUE_RUNNING_EXIT_CALLBACK";
        break;
    case JOB_QUEUE_STATUS_FAILURE:
        return "JOB_QUEUE_STATUS_FAILURE";
        break;
    case JOB_QUEUE_FAILED:
        return "JOB_QUEUE_FAILED";
        break;
    case JOB_QUEUE_DO_KILL_NODE_FAILURE:
        return "JOB_QUEUE_DO_KILL_NODE_FAILURE";
        break;
    case JOB_QUEUE_UNKNOWN:
        return "JOB_QUEUE_UNKNOWN";
        break;
    }

    util_abort("%s: internal error", __func__);
    return NULL;
}
