#ifndef ERT_JOB_QUEUE_H
#define ERT_JOB_QUEUE_H
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#include <ert/job_queue/job_node.hpp>
#include <ert/job_queue/queue_driver.hpp>

typedef struct job_queue_struct job_queue_type;
extern "C" PY_USED void job_queue_submit_complete(job_queue_type *queue);
extern "C" void job_queue_set_driver(job_queue_type *queue,
                                     queue_driver_type *driver);
extern "C" job_queue_type *job_queue_alloc(int, const char *status_file,
                                           const char *exit_file);
extern "C" void job_queue_free(job_queue_type *);

int job_queue_add_job(job_queue_type *, const char *run_cmd,
                      job_callback_ftype *done_callback,
                      job_callback_ftype *retry_callback,
                      job_callback_ftype *exit_callback, void *callback_arg,
                      int num_cpu, const char *, const char *, int argc,
                      const char **argv);

bool job_queue_accept_jobs(const job_queue_type *queue);

int job_queue_iget_status_summary(const job_queue_type *queue,
                                  job_status_type status);

extern "C" PY_USED void
job_queue_set_max_job_duration(job_queue_type *queue, int max_duration_seconds);
extern "C" bool job_queue_kill_job(job_queue_type *queue, int job_index);
extern "C" PY_USED bool job_queue_is_running(const job_queue_type *queue);
extern "C" int job_queue_get_max_submit(const job_queue_type *job_queue);
extern "C" int job_queue_get_num_running(const job_queue_type *queue);
extern "C" int job_queue_get_num_pending(const job_queue_type *queue);
extern "C" int job_queue_get_num_waiting(const job_queue_type *queue);
extern "C" int job_queue_get_num_complete(const job_queue_type *queue);
extern "C" PY_USED void *job_queue_iget_driver_data(job_queue_type *queue,
                                                    int job_index);

extern "C" PY_USED char *job_queue_get_exit_file(const job_queue_type *queue);
extern "C" PY_USED char *job_queue_get_status_file(const job_queue_type *queue);
extern "C" PY_USED int job_queue_add_job_node(job_queue_type *queue,
                                              job_queue_node_type *node);

#endif
