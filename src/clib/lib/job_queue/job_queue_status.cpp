#include <pthread.h>

#include <ert/job_queue/job_queue_status.hpp>
#include <ert/job_queue/queue_driver.hpp>

job_queue_status_type *job_queue_status_alloc() {
    auto job_queue = new job_queue_status_struct;
    pthread_rwlock_init(&job_queue->rw_lock, nullptr);
    return job_queue;
}

void job_queue_status_free(job_queue_status_type *status) { delete status; }

int job_queue_status_get_count(job_queue_status_type *status_count,
                               int job_status_mask) {
    int count = 0, index = 0, status = 1;
    pthread_rwlock_rdlock(&status_count->rw_lock);
    {
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

    for (int index = 0; index < JOB_QUEUE_MAX_STATE; index++) {
        if (status_count->status_index[index] == status_type) {
            pthread_rwlock_wrlock(&status_count->rw_lock);
            {
                int count = status_count->status_list[index];
                status_count->status_list[index] = count + step;
            }
            pthread_rwlock_unlock(&status_count->rw_lock);
            break;
        }
    }
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