#include <ert/job_queue/local_driver.hpp>
#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>
#include <ert/job_queue/torque_driver.hpp>
#include <ert/util/util.hpp>

/*
   This file implements the datatype queue_driver_type which is an
   abstract datatype for communicating with a subsystem for
   communcating with other low-level systems for running external
   jobs. The job_queue instance, which will handle a queue of jobs,
   interacts with the jobs through a queue_driver instance.

   The queue_driver type is a quite small datastructure which "wraps"
   and underlying specific driver instance; examples of specific
   driver instances are the lsf_driver which communicates with the LSF
   system and the local_driver which runs jobs directly on the current
   workstation. The queue_driver type contains essentially three
   different types of fields:

    1. Functions pointers for manipulating the jobs, and the state of
       the low-level driver.

    2. An opaque (i.e. void *) pointer to the state of the low level
       driver. This will be passed as first argument to all the
       function pointers, e.g. like the "self" in Python methods.

    3. Some data fields which are common to all driver types.

 */

struct queue_driver_struct {
    /** Function pointers - pointing to low level functions in the
     * implementations of e.g. lsf_driver. */
    submit_job_ftype *submit = nullptr;
    free_job_ftype *free_job = nullptr;
    kill_job_ftype *kill_job = nullptr;
    get_status_ftype *get_status = nullptr;
    free_queue_driver_ftype *free_driver = nullptr;
    set_option_ftype *set_option = nullptr;
    get_option_ftype *get_option = nullptr;
    init_option_list_ftype *init_options = nullptr;

    /** Driver specific data - passed as first argument to the driver functions above. */
    void *data = nullptr;
};

bool queue_driver_set_option(queue_driver_type *driver, const char *option_key,
                             const void *value) {
    return driver->set_option(driver->data, option_key, value);
}

/**
   Observe that after the driver instance has been allocated it does
   NOT support modification of the common fields, only the data owned
   by the specific low level driver, i.e. the LSF data, can be
   modified runtime.

   The driver returned from the queue_driver_alloc_empty() function is
   NOT properly initialized and NOT ready for use.
 */
static queue_driver_type *queue_driver_alloc_empty() {
    return new queue_driver_type;
}

// The driver created in this function has all the function pointers
// correctly initialized; but no options have been set. I.e. unless
// the driver in question needs no options (e.g. the LOCAL driver) the
// returned driver will NOT be ready for use.

queue_driver_type *queue_driver_alloc(job_driver_type type) {
    queue_driver_type *driver = queue_driver_alloc_empty();
    switch (type) {
    case LSF_DRIVER:
        driver->submit = lsf_driver_submit_job;
        driver->get_status = lsf_driver_get_job_status;
        driver->kill_job = lsf_driver_kill_job;
        driver->free_job = lsf_driver_free_job;
        driver->free_driver = lsf_driver_free__;
        driver->set_option = lsf_driver_set_option;
        driver->get_option = lsf_driver_get_option;
        driver->init_options = lsf_driver_init_option_list;
        driver->data = lsf_driver_alloc();
        break;
    case LOCAL_DRIVER:
        driver->submit = local_driver_submit_job;
        driver->get_status = local_driver_get_job_status;
        driver->kill_job = local_driver_kill_job;
        driver->free_job = local_driver_free_job;
        driver->free_driver = local_driver_free__;
        driver->set_option = local_driver_set_option;
        driver->get_option = local_driver_get_option;
        driver->init_options = local_driver_init_option_list;
        driver->data = local_driver_alloc();
        break;
    case TORQUE_DRIVER:
        driver->submit = torque_driver_submit_job;
        driver->get_status = torque_driver_get_job_status;
        driver->kill_job = torque_driver_kill_job;
        driver->free_job = torque_driver_free_job;
        driver->free_driver = torque_driver_free__;
        driver->set_option = torque_driver_set_option;
        driver->get_option = torque_driver_get_option;
        driver->init_options = torque_driver_init_option_list;
        driver->data = torque_driver_alloc();
        break;
    case SLURM_DRIVER:
        driver->set_option = slurm_driver_set_option;
        driver->get_option = slurm_driver_get_option;
        driver->init_options = slurm_driver_init_option_list;
        driver->free_driver = slurm_driver_free__;
        driver->kill_job = slurm_driver_kill_job;
        driver->free_job = slurm_driver_free_job;
        driver->submit = slurm_driver_submit_job;
        driver->get_status = slurm_driver_get_job_status;
        driver->data = slurm_driver_alloc();
        break;
    default:
        util_abort("%s: unrecognized driver type:%d \n", __func__, type);
    }

    return driver;
}

const void *queue_driver_get_option(queue_driver_type *driver,
                                    const char *option_key) {
    return driver->get_option(driver->data, option_key);
}

void queue_driver_init_option_list(queue_driver_type *driver,
                                   stringlist_type *option_list) {
    driver->init_options(option_list);
}

/* These are the functions used by the job_queue layer. */

void *queue_driver_submit_job(queue_driver_type *driver, const char *run_cmd,
                              int num_cpu, const char *run_path,
                              const char *job_name, int argc,
                              const char **argv) {
    return driver->submit(driver->data, run_cmd, num_cpu, run_path, job_name,
                          argc, argv);
}

void queue_driver_free_job(queue_driver_type *driver, void *job_data) {
    driver->free_job(job_data);
}

void queue_driver_kill_job(queue_driver_type *driver, void *job_data) {
    driver->kill_job(driver->data, job_data);
}

job_status_type queue_driver_get_status(queue_driver_type *driver,
                                        void *job_data) {
    job_status_type status = driver->get_status(driver->data, job_data);
    return status;
}

void queue_driver_free_driver(queue_driver_type *driver) {
    driver->free_driver(driver->data);
}

void queue_driver_free(queue_driver_type *driver) {
    queue_driver_free_driver(driver);
    delete driver;
}
