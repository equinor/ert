#ifndef ERT_JOB_NODE_H
#define ERT_JOB_NODE_H

#include <ert/job_queue/job_queue_status.hpp>
#include <ert/job_queue/queue_driver.hpp>

/**
   This struct holds the job_queue information about one job. Observe
   the following:

    1. This struct is purely static - i.e. it is invisible outside of
       this file-scope.

    2. Typically the driver would like to store some additional
       information, i.e. the PID of the running process for the local
       driver; that is stored in a (driver specific) struct under the
       field job_data.

    3. If the driver detects that a job has failed it leaves an EXIT
       file, the exit status is (currently) not reliably transferred
       back to to the job_queue layer.

*/

typedef bool(job_callback_ftype)(void *);
typedef struct job_queue_node_struct job_queue_node_type;

bool job_queue_node_status_transition(job_queue_node_type *node,
                                      job_queue_status_type *status,
                                      job_status_type new_status);
submit_status_type job_queue_node_submit(job_queue_node_type *node,
                                         job_queue_status_type *status,
                                         queue_driver_type *driver);
extern "C" PY_USED submit_status_type job_queue_node_submit_simple(
    job_queue_node_type *node, queue_driver_type *driver);
void job_queue_node_free_error_info(job_queue_node_type *node);
void job_queue_node_fscanf_EXIT(job_queue_node_type *node);
void job_queue_node_free_data(job_queue_node_type *node);
job_queue_node_type *
job_queue_node_alloc(const char *job_name, const char *run_path,
                     const char *run_cmd, int argc, char const *const *argv,
                     int num_cpu, const char *status_file,
                     const char *exit_file, job_callback_ftype *done_callback,
                     job_callback_ftype *retry_callback,
                     job_callback_ftype *exit_callback, void *callback_arg);

job_queue_node_type *job_queue_node_alloc_simple(const char *job_name,
                                                 const char *run_path,
                                                 const char *run_cmd, int argc,
                                                 const char **argv);

extern "C" PY_USED job_queue_node_type *
job_queue_node_alloc_python(const char *job_name, const char *run_path,
                            const char *run_cmd, int argc,
                            const stringlist_type *arguments, int num_cpu,
                            const char *status_file, const char *exit_file);

bool job_queue_node_kill(job_queue_node_type *node,
                         job_queue_status_type *status,
                         queue_driver_type *driver);
extern "C" PY_USED bool job_queue_node_kill_simple(job_queue_node_type *node,
                                                   queue_driver_type *driver);
extern "C" void job_queue_node_free(job_queue_node_type *node);
extern "C" job_status_type
job_queue_node_get_status(const job_queue_node_type *node);
extern "C" PY_USED job_status_type job_queue_node_refresh_status(
    job_queue_node_type *node, queue_driver_type *driver);
extern "C" int
job_queue_node_get_submit_attempt(const job_queue_node_type *node);

double job_queue_node_time_since_sim_start(const job_queue_node_type *node);

int job_queue_node_get_queue_index(const job_queue_node_type *node);
void job_queue_node_set_queue_index(job_queue_node_type *node, int queue_index);

void *job_queue_node_get_driver_data(job_queue_node_type *node);
extern "C" void job_queue_node_set_status(job_queue_node_type *node,
                                          job_status_type new_status);

#endif
