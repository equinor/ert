#ifndef ERT_SLURM_DRIVER_H
#define ERT_SLURM_DRIVER_H

#include <string>
#include <vector>

#include <ert/job_queue/job_status.hpp>

/*
  The options supported by the Slurm driver; these string constants will be used
  in the user configuration file - i.e. they are very much part of API and
  remain stable.
*/

typedef struct slurm_driver_struct slurm_driver_type;

#define SLURM_SBATCH_OPTION "SBATCH"
#define SLURM_SCANCEL_OPTION "SCANCEL"
#define SLURM_SCONTROL_OPTION "SCONTROL"
#define SLURM_SQUEUE_OPTION "SQUEUE"
#define SLURM_PARTITION_OPTION "PARTITION"
#define SLURM_SQUEUE_TIMEOUT_OPTION "SQUEUE_TIMEOUT"

// Observe that the SLURM_MAX_RUNTIME_OPTION expects a time limit in seconds,
// whereas slurm uses a time limit in minutes
#define SLURM_MAX_RUNTIME_OPTION "MAX_RUNTIME"
#define SLURM_MEMORY_OPTION "MEMORY"
#define SLURM_MEMORY_PER_CPU_OPTION "MEMORY_PER_CPU"

// For the EXCLUDE and INCLUDE host options the slurm driver
// maintains an internal list of hostnames, and the option can be called
// repeatedly. It is possible to add multiple hosts separated by space or comma
// in one option call:
//
// QUEUE_OPTION SLURM EXCLUDE_HOST host1,host2,host3
// QUEUE_OPTION SLURM EXCLUDE_HOST host5 host6,host7
#define SLURM_EXCLUDE_HOST_OPTION "EXCLUDE_HOST"
#define SLURM_INCLUDE_HOST_OPTION "INCLUDE_HOST"

const std::vector<std::string> SLURM_DRIVER_OPTIONS = {
    SLURM_SBATCH_OPTION,         SLURM_SCONTROL_OPTION,
    SLURM_SQUEUE_OPTION,         SLURM_SCANCEL_OPTION,
    SLURM_PARTITION_OPTION,      SLURM_SQUEUE_TIMEOUT_OPTION,
    SLURM_MAX_RUNTIME_OPTION,    SLURM_MEMORY_OPTION,
    SLURM_MEMORY_PER_CPU_OPTION, SLURM_INCLUDE_HOST_OPTION,
    SLURM_EXCLUDE_HOST_OPTION};

void *slurm_driver_alloc();
void slurm_driver_free(slurm_driver_type *driver);
void slurm_driver_free__(void *__driver);
const void *slurm_driver_get_option(const void *__driver,
                                    const char *option_key);
bool slurm_driver_set_option(void *__driver, const char *option_key,
                             const void *value);
void *slurm_driver_submit_job(void *__driver, const char *cmd, int num_cpu,
                              const char *run_path, const char *job_name);
job_status_type slurm_driver_get_job_status(void *__driver, void *__job);
void slurm_driver_kill_job(void *__driver, void *__job);
void slurm_driver_free_job(void *__job);

#endif
