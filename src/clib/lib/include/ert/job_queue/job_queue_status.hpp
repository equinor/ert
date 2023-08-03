#ifndef ERT_JOB_QUEUE_STATUS_H
#define ERT_JOB_QUEUE_STATUS_H

#include <time.h>

#include <ert/job_queue/queue_driver.hpp>

typedef struct job_queue_status_struct job_queue_status_type;

job_queue_status_type *job_queue_status_alloc();
void job_queue_status_free(job_queue_status_type *status);
int job_queue_status_get_count(job_queue_status_type *status,
                               int job_status_mask);
void job_queue_status_step(job_queue_status_type *status_count,
                           job_status_type status_type, int step);
bool job_queue_status_transition(job_queue_status_type *status_count,
                                 job_status_type src_status,
                                 job_status_type target_status);
int job_queue_status_get_total_count(const job_queue_status_type *status);

#endif
