#ifndef ERT_LOCAL_DRIVER_H
#define ERT_LOCAL_DRIVER_H

#include <ert/job_queue/queue_driver.hpp>

typedef struct local_driver_struct local_driver_type;

void *local_driver_alloc();

void *local_driver_submit_job(void *_driver, const char *submit_cmd,
                              int num_cpu, const char *run_path,
                              const char *job_name);
void local_driver_kill_job(void *_driver, void *_job);
void local_driver_free_(void *_driver);
job_status_type local_driver_get_job_status(void *_driver, void *_job);
void local_driver_free_job(void *_job);
bool local_driver_set_option(void *_driver, const char *option_key,
                             const void *value_);
const void *local_driver_get_option(const void *_driver,
                                    const char *option_key);
#endif
