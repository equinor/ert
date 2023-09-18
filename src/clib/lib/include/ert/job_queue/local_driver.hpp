#ifndef ERT_LOCAL_DRIVER_H
#define ERT_LOCAL_DRIVER_H

#include <ert/job_queue/queue_driver.hpp>

typedef struct local_driver_struct local_driver_type;

void *local_driver_alloc();

void *local_driver_submit_job(void *__driver, const char *submit_cmd,
                              int num_cpu, const char *run_path,
                              const char *job_name, int argc,
                              const char **argv);
void local_driver_kill_job(void *__driver, void *__job);
void local_driver_free__(void *__driver);
job_status_type local_driver_get_job_status(void *__driver, void *__job);
void local_driver_free_job(void *__job);
void local_driver_init_option_list(stringlist_type *option_list);
bool local_driver_set_option(void *__driver, const char *option_key,
                             const void *value_);
const void *local_driver_get_option(const void *__driver,
                                    const char *option_key);
#endif
