#include <pthread.h>

#include <ert/job_queue/job_queue_status.hpp>
#include <ert/job_queue/queue_driver.hpp>

struct job_queue_status_struct {
    int status_list[JOB_QUEUE_MAX_STATE];
    pthread_rwlock_t rw_lock;
    int status_index[JOB_QUEUE_MAX_STATE];
};

job_queue_status_type *job_queue_status_alloc() {
    job_queue_status_type *status =
        (job_queue_status_type *)util_malloc(sizeof *status);
    pthread_rwlock_init(&status->rw_lock, NULL);

    for (int &status_list_value : status->status_list)
        status_list_value = 0;

    status->status_index[0] =
        JOB_QUEUE_NOT_ACTIVE; // Initial, allocated job state, job not added - controlled by job_queue
    status->status_index[1] =
        JOB_QUEUE_WAITING; // The job is ready to be started - controlled by job_queue
    status->status_index[2] =
        JOB_QUEUE_SUBMITTED; // Job is submitted to driver - temporary state - controlled by job_queue
    status->status_index[3] =
        JOB_QUEUE_PENDING; // Job is pending, before actual execution - controlled by queue_driver
    status->status_index[4] =
        JOB_QUEUE_RUNNING; // Job is executing - controlled by queue_driver
    status->status_index[5] =
        JOB_QUEUE_DONE; // Job is done (successful or not), temporary state - controlled/returned by by queue_driver
    status->status_index[6] =
        JOB_QUEUE_EXIT; //Job is done, with exit status != 0, temporary state - controlled/returned by by queue_driver
    status->status_index[7] =
        JOB_QUEUE_IS_KILLED; // Job has been killed, due to JOB_QUEUE_DO_KILL, FINAL STATE - controlled by job_queue
    status->status_index[8] =
        JOB_QUEUE_DO_KILL; // User / queue system has requested killing of job - controlled by job_queue / external scope
    status->status_index[9] =
        JOB_QUEUE_SUCCESS; // All good, comes after JOB_QUEUE_DONE, with additional checks, FINAL STATE - controlled by job_queue
    status->status_index[10] =
        JOB_QUEUE_RUNNING_DONE_CALLBACK; // Temporary state, while running requested callbacks after an ended job - controlled by job_queue
    status->status_index[11] =
        JOB_QUEUE_RUNNING_EXIT_CALLBACK; // Temporary state, while running requested callbacks after an ended job - controlled by job_queue
    status->status_index[12] =
        JOB_QUEUE_STATUS_FAILURE; //The driver call to get status has failed, job status remains unchanged
    status->status_index[13] =
        JOB_QUEUE_FAILED; // Job has failed, no more retries, FINAL STATE
    status->status_index[14] =
        JOB_QUEUE_DO_KILL_NODE_FAILURE; // Job has failed, node should be blacklisted
    status->status_index[15] =
        JOB_QUEUE_UNKNOWN; // Unable to get status from submitted job

    return status;
}

void job_queue_status_free(job_queue_status_type *status) { free(status); }

int job_queue_status_get_count(job_queue_status_type *status_count,
                               int job_status_mask) {
    int count = 0;
    pthread_rwlock_rdlock(&status_count->rw_lock);
    {
        int index = 0;
        int status = 1;

        while (true) {
            if ((status & job_status_mask) == status) {
                job_status_mask -= status;
                count += status_count->status_list[index];
            }

            if (job_status_mask == 0)
                break;

            index++;
            status <<= 1;
            if (index == JOB_QUEUE_MAX_STATE)
                util_abort("%s: internal error: remaining unrecognized status "
                           "value:%d \n",
                           __func__, job_status_mask);
        }
    }
    pthread_rwlock_unlock(&status_count->rw_lock);
    return count;
}

void job_queue_status_step(job_queue_status_type *status_count,
                           job_status_type status_type, int step) {
    int status_index = -1;

    for (int index = 0; index < JOB_QUEUE_MAX_STATE; index++) {
        if (status_count->status_index[index] == status_type) {
            status_index = index;
            break;
        }
    }

    if (status_index == -1)
        util_abort("%s: failed to get index from status:%d \n", __func__,
                   status_type);

    pthread_rwlock_wrlock(&status_count->rw_lock);
    {
        int count = status_count->status_list[status_index];
        status_count->status_list[status_index] = count + step;
    }
    pthread_rwlock_unlock(&status_count->rw_lock);
}

/*
  Each individual ++ and -- operation is atomic, if the different status
  counts do not add up perfectly at all times that is ok.
*/
bool job_queue_status_transition(job_queue_status_type *status_count,
                                 job_status_type src_status,
                                 job_status_type target_status) {

    // Leave current status as is in case of JOB_QUEUE_STATUS_FAILURE
    if ((src_status == target_status) ||
        (target_status == JOB_QUEUE_STATUS_FAILURE))
        return false;

    job_queue_status_step(status_count, src_status, -1);
    job_queue_status_step(status_count, target_status, 1);

    return true;
}

int job_queue_status_get_total_count(const job_queue_status_type *status) {
    int total_count = 0;
    for (int value : status->status_list)
        total_count += value;
    return total_count;
}
