#ifndef ERT_JOB_QUEUE_H
#define ERT_JOB_QUEUE_H
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#include <ert/job_queue/job_node.hpp>
#include <ert/job_queue/queue_driver.hpp>

typedef struct job_queue_struct job_queue_type;
extern "C" job_queue_type *job_queue_alloc(queue_driver_type *driver);
extern "C" void job_queue_free(job_queue_type *);
extern "C" PY_USED int job_queue_add_job_node(job_queue_type *queue,
                                              job_queue_node_type *node);
#endif
