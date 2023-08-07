#ifndef TORQUE_DRIVER_H
#define TORQUE_DRIVER_H

#include <stdio.h>
#include <string>
#include <vector>

#include <ert/job_queue/queue_driver.hpp>

/* The options supported by the Torque driver. */
#define TORQUE_QSUB_CMD "QSUB_CMD"
#define TORQUE_QSTAT_CMD "QSTAT_CMD"
#define TORQUE_QSTAT_OPTIONS "QSTAT_OPTIONS"
#define TORQUE_QDEL_CMD "QDEL_CMD"
#define TORQUE_QUEUE "QUEUE"
#define TORQUE_NUM_CPUS_PER_NODE "NUM_CPUS_PER_NODE"
#define TORQUE_NUM_NODES "NUM_NODES"
#define TORQUE_KEEP_QSUB_OUTPUT "KEEP_QSUB_OUTPUT"
#define TORQUE_CLUSTER_LABEL "CLUSTER_LABEL"
#define TORQUE_JOB_PREFIX_KEY "JOB_PREFIX"
#define TORQUE_SUBMIT_SLEEP "SUBMIT_SLEEP"
#define TORQUE_DEBUG_OUTPUT "DEBUG_OUTPUT"
#define TORQUE_QUEUE_QUERY_TIMEOUT "QUEUE_QUERY_TIMEOUT"

#define TORQUE_DEFAULT_QSUB_CMD "qsub"
#define TORQUE_DEFAULT_QSTAT_CMD "qstat_proxy.sh"
#define TORQUE_DEFAULT_QSTAT_OPTIONS "-x"
#define TORQUE_DEFAULT_QDEL_CMD "qdel"
#define TORQUE_DEFAULT_SUBMIT_SLEEP "0"
#define TORQUE_DEFAULT_QUEUE_QUERY_TIMEOUT "126"

typedef struct torque_driver_struct torque_driver_type;
typedef struct torque_job_struct torque_job_type;

const std::vector<std::string> TORQUE_DRIVER_OPTIONS = {
    TORQUE_QSUB_CMD,
    TORQUE_QSTAT_CMD,
    TORQUE_QSTAT_OPTIONS,
    TORQUE_QDEL_CMD,
    TORQUE_QUEUE,
    TORQUE_NUM_CPUS_PER_NODE,
    TORQUE_NUM_NODES,
    TORQUE_KEEP_QSUB_OUTPUT,
    TORQUE_CLUSTER_LABEL,
    TORQUE_JOB_PREFIX_KEY,
    TORQUE_QUEUE_QUERY_TIMEOUT};

void *torque_driver_alloc();

void *torque_driver_submit_job(void *__driver, const char *submit_cmd,
                               int num_cpu, const char *run_path,
                               const char *job_name, int argc,
                               const char **argv);

void torque_driver_kill_job(void *__driver, void *__job);
void torque_driver_free__(void *__driver);
void torque_driver_free(torque_driver_type *driver);
job_status_type torque_driver_get_job_status(void *__driver, void *__job);
void torque_driver_free_job(void *__job);

const void *torque_driver_get_option(const void *__driver,
                                     const char *option_key);
bool torque_driver_set_option(void *__driver, const char *option_key,
                              const void *value);
void torque_driver_init_option_list(stringlist_type *option_list);

void torque_job_create_submit_script(const char *run_path,
                                     const char *submit_cmd, int argc,
                                     const char *const *job_argv);
int torque_driver_get_submit_sleep(const torque_driver_type *driver);
int torque_driver_get_timeout(const torque_driver_type *driver);
FILE *torque_driver_get_debug_stream(const torque_driver_type *driver);
job_status_type torque_driver_parse_status(const char *qstat_file,
                                           const char *jobnr);

#endif /* TORQUE_DRIVER_H */
