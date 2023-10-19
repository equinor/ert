#ifndef ERT_JOB_NODE_H
#define ERT_JOB_NODE_H

#include <ert/util/stringlist.hpp>
#include <optional>
#include <string.h>

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
struct job_queue_node_struct {
    /** How many cpu's will this job need - the driver is free to ignore if not relevant. */
    int num_cpu = 0;
    /** The path to the actual executable. */
    char *run_cmd = nullptr;
    /** The queue will look for the occurrence of this file to detect a failure. */
    char *exit_file = nullptr;
    /** The queue will look for this file to verify that the job is running or
     * has run. */
    char *status_file = nullptr;
    /** The name of the job. */
    char *job_name = nullptr;
    /** Where the job is run - absolute path. */
    char *run_path = nullptr;
    int queue_index = 0;
    bool confirmed_running = false;

    std::optional<std::string> fail_message{};

    /** Which attempt is this ... */
    int submit_attempt = 0;
    /** The current status of the job. */
    job_status_type job_status = JOB_QUEUE_NOT_ACTIVE;
    /** Protecting the access to the job_data pointer. */
    pthread_mutex_t data_mutex;
    /** Driver specific data about this job - fully handled by the driver. */
    void *job_data = nullptr;
    /** When did the job change status -> RUNNING - the LAST TIME. */
    time_t sim_start = 0;
};

typedef bool(job_callback_ftype)(void *);
typedef struct job_queue_node_struct job_queue_node_type;

void job_queue_node_fscanf_EXIT(job_queue_node_type *node);
void job_queue_node_free_data(job_queue_node_type *node);

extern "C" PY_USED job_queue_node_type *
job_queue_node_alloc(const char *job_name, const char *run_path,
                     const char *run_cmd, int num_cpu, const char *status_file,
                     const char *exit_file);

extern "C" void job_queue_node_free(job_queue_node_type *node);
extern "C" job_status_type
job_queue_node_get_status(const job_queue_node_type *node);

int job_queue_node_get_queue_index(const job_queue_node_type *node);
void job_queue_node_set_queue_index(job_queue_node_type *node, int queue_index);

extern "C" void job_queue_node_set_status(job_queue_node_type *node,
                                          job_status_type new_status);

#endif
