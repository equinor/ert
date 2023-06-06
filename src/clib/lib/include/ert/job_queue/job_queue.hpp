#ifndef ERT_JOB_QUEUE_H
#define ERT_JOB_QUEUE_H
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#include <ert/job_queue/job_node.hpp>
#include <ert/job_queue/queue_driver.hpp>

typedef struct job_queue_struct job_queue_type;
extern "C" void job_queue_set_driver(job_queue_type *queue,
                                     queue_driver_type *driver);
extern "C" job_queue_type *job_queue_alloc();
extern "C" void job_queue_free(job_queue_type *);

int job_queue_iget_status_summary(const job_queue_type *queue,
                                  job_status_type status);

extern "C" int job_queue_get_num_running(const job_queue_type *queue);
extern "C" int job_queue_get_num_pending(const job_queue_type *queue);
extern "C" int job_queue_get_num_waiting(const job_queue_type *queue);
extern "C" int job_queue_get_num_complete(const job_queue_type *queue);
extern "C" PY_USED void *job_queue_iget_driver_data(job_queue_type *queue,
                                                    int job_index);

extern "C" PY_USED int job_queue_add_job_node(job_queue_type *queue,
                                              job_queue_node_type *node);

#endif
