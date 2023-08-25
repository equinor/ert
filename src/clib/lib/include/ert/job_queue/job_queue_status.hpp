#ifndef ERT_JOB_QUEUE_STATUS_H
#define ERT_JOB_QUEUE_STATUS_H

#include <time.h>

#include <ert/job_queue/queue_driver.hpp>

typedef struct job_queue_status_struct job_queue_status_type;

struct job_queue_status_struct {
    int status_list[JOB_QUEUE_MAX_STATE] = {0};
    pthread_rwlock_t rw_lock{};

    const int status_index[JOB_QUEUE_MAX_STATE] = {
        JOB_QUEUE_NOT_ACTIVE, // Initial, allocated job state, job not added - controlled by job_queue
        JOB_QUEUE_WAITING, // The job is ready to be started - controlled by job_queue
        JOB_QUEUE_SUBMITTED, // Job is submitted to driver - temporary state - controlled by job_queue
        JOB_QUEUE_PENDING, // Job is pending, before actual execution - controlled by queue_driver
        JOB_QUEUE_RUNNING, // Job is executing - controlled by queue_driver
        JOB_QUEUE_DONE, // Job is done (successful or not), temporary state - controlled/returned by by queue_driver
        JOB_QUEUE_EXIT, //Job is done, with exit status != 0, temporary state - controlled/returned by by queue_driver
        JOB_QUEUE_IS_KILLED, // Job has been killed, due to JOB_QUEUE_DO_KILL, FINAL STATE - controlled by job_queue
        JOB_QUEUE_DO_KILL, // User / queue system has requested killing of job - controlled by job_queue / external scope
        JOB_QUEUE_SUCCESS, // All good, comes after JOB_QUEUE_DONE, with additional checks, FINAL STATE - controlled by job_queue
        JOB_QUEUE_RUNNING_DONE_CALLBACK, // Temporary state, while running requested callbacks after an ended job - controlled by job_queue
        JOB_QUEUE_RUNNING_EXIT_CALLBACK, // Temporary state, while running requested callbacks after an ended job - controlled by job_queue
        JOB_QUEUE_STATUS_FAILURE, // The driver call to get status has failed, job status remains unchanged
        JOB_QUEUE_FAILED, // Job has failed, no more retries, FINAL STATE
        JOB_QUEUE_UNKNOWN}; // Unable to get status from submitted job
};

job_queue_status_type *job_queue_status_alloc();
void job_queue_status_free(job_queue_status_type *status);
int job_queue_status_get_count(job_queue_status_type *status,
                               int job_status_mask);
void job_queue_status_step(job_queue_status_type *status_count,
                           job_status_type status_type, int step);
bool job_queue_status_transition(job_queue_status_type *status_count,
                                 job_status_type src_status,
                                 job_status_type target_status);
#endif
