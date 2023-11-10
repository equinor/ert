#pragma once
#include <ert/job_queue/job_status.hpp>

using job_driver_type = enum {
    NULL_DRIVER = 0,
    LSF_DRIVER = 1,
    LOCAL_DRIVER = 2,
    TORQUE_DRIVER = 4,
    SLURM_DRIVER = 5
};

using queue_driver_type = struct queue_driver_struct;

using submit_job_ftype = void *(void *, const char *, int, const char *,
                                const char *);
using kill_job_ftype = void(void *, void *);
using get_status_ftype = job_status_type(void *, void *);
using free_job_ftype = void(void *);
using free_queue_driver_ftype = void(void *);
using set_option_ftype = bool(void *, const char *, const void *);
using get_option_ftype = const void *(const void *, const char *);

extern "C" queue_driver_type *queue_driver_alloc(job_driver_type type);

void *queue_driver_submit_job(queue_driver_type *driver, const char *run_cmd,
                              int num_cpu, const char *run_path,
                              const char *job_name);
void queue_driver_free_job(queue_driver_type *driver, void *job_data);
void queue_driver_kill_job(queue_driver_type *driver, void *job_data);
job_status_type queue_driver_get_status(queue_driver_type *driver,
                                        void *job_data);
extern "C" bool queue_driver_set_option(queue_driver_type *driver,
                                        const char *option_key,
                                        const void *value);
extern "C" const void *queue_driver_get_option(queue_driver_type *driver,
                                               const char *option_key);
extern "C" void queue_driver_free(queue_driver_type *driver);

using submit_status_type = enum {
    SUBMIT_OK = 0,
    /** Typically no more attempts. */
    SUBMIT_JOB_FAIL = 1,
    /** The driver would not take the job - for whatever reason?? */
    SUBMIT_DRIVER_FAIL = 2,
    /** The queue is currently not accepting more jobs
     * - either (temporarilty) because of pause or it is going down. */
    SUBMIT_QUEUE_CLOSED = 3
};
