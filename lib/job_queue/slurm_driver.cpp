/*
   Copyright (C) 2020  Equinor ASA, Norway.

   The file 'slurm_driver.cpp' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>

#include <string>
#include <vector>

#include <ert/util/util.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/res_util/log.hpp>
#include <ert/res_util/res_log.hpp>
#include <ert/res_util/res_env.hpp>

#include <ert/job_queue/queue_driver.hpp>
#include <ert/job_queue/slurm_driver.hpp>


class SlurmJob {
};


#define SLURM_DRIVER_TYPE_ID  70555081
#define DEFAULT_SBATCH_CMD    "sbatch"
#define DEFAULT_SCANCEL_CMD   "scancel"
#define DEFAULT_SCONTROL_CMD     "scontrol"
#define DEFAULT_SQUEUE_CMD    "sqeueue"

struct slurm_driver_struct {
  UTIL_TYPE_ID_DECLARATION;

  std::string sbatch_cmd;
  std::string scancel_cmd;
  std::string squeue_cmd;
  std::string scontrol_cmd;
  std::string partition;
};

static std::string load_file(const char * fname) {
    char * buffer = util_fread_alloc_file_content(fname, nullptr);
    std::string s = buffer;
    free(buffer);
    return s;
}


static std::string load_stdout(const char * cmd, int argc, const char ** argv) {
    std::string fname = std::string(cmd) + "-stdout";
    char * stdout = (char*) util_alloc_tmp_file("/tmp", fname.c_str(), true);

    auto exit_status = util_spawn_blocking(cmd, argc, argv, stdout, nullptr);
    auto file_content = load_file(stdout);

    if (exit_status != 0)
      res_log_fwarning("Calling shell command %s ... returned non zero exitcode: %d", cmd, exit_status);

    util_unlink_existing(stdout);
    free(stdout);
    return file_content;
}


static std::string load_stdout(const char * cmd, const std::vector<std::string>& args) {
    const char ** argv = static_cast<const char **>(util_calloc( args.size(), sizeof * argv ));
    for (std::size_t i=0; i < args.size(); i++)
        argv[i] = args[i].c_str();

    auto file_content = load_stdout(cmd, args.size(), argv);
    free(argv);
    return file_content;
}



UTIL_SAFE_CAST_FUNCTION( slurm_driver , SLURM_DRIVER_TYPE_ID)
static UTIL_SAFE_CAST_FUNCTION_CONST( slurm_driver , SLURM_DRIVER_TYPE_ID)

void * slurm_driver_alloc() {
  slurm_driver_type * driver = new slurm_driver_type();
  UTIL_TYPE_ID_INIT(driver, SLURM_DRIVER_TYPE_ID);
  driver->sbatch_cmd = DEFAULT_SBATCH_CMD;
  driver->scancel_cmd = DEFAULT_SCANCEL_CMD;
  driver->squeue_cmd = DEFAULT_SQUEUE_CMD;
  driver->scontrol_cmd = DEFAULT_SCONTROL_CMD;
  return driver;
}

void slurm_driver_free(slurm_driver_type * driver) {
  delete driver;
}

void slurm_driver_free__(void * __driver ) {
  slurm_driver_type * driver = slurm_driver_safe_cast( __driver );
  slurm_driver_free( driver );
}


const void * slurm_driver_get_option( const void * __driver, const char * option_key) {
  const slurm_driver_type * driver = slurm_driver_safe_cast_const( __driver );
  if (strcmp(option_key, SLURM_SBATCH_OPTION) == 0)
    return driver->sbatch_cmd.c_str();

  if (strcmp(option_key, SLURM_SCANCEL_OPTION) == 0)
    return driver->scancel_cmd.c_str();

  if (strcmp(option_key, SLURM_SCONTROL_OPTION) == 0)
    return driver->scontrol_cmd.c_str();

  if (strcmp(option_key, SLURM_SQUEUE_OPTION) == 0)
    return driver->squeue_cmd.c_str();

  if (strcmp(option_key, SLURM_PARTITION_OPTION) == 0)
    return driver->partition.c_str();

  return nullptr;
}


bool slurm_driver_set_option( void * __driver, const char * option_key, const void * value) {
  slurm_driver_type * driver = slurm_driver_safe_cast( __driver );
  if (strcmp(option_key, SLURM_SBATCH_OPTION) == 0) {
    driver->sbatch_cmd = static_cast<const char*>(value);
    return true;
  }

  if (strcmp(option_key, SLURM_SCANCEL_OPTION) == 0) {
    driver->scancel_cmd = static_cast<const char*>(value);
    return true;
  }

  if (strcmp(option_key, SLURM_SQUEUE_OPTION) == 0) {
    driver->squeue_cmd = static_cast<const char*>(value);
    return true;
  }

  if (strcmp(option_key, SLURM_SCONTROL_OPTION) == 0) {
    driver->scontrol_cmd = static_cast<const char*>(value);
    return true;
  }

  if (strcmp(option_key, SLURM_PARTITION_OPTION) == 0) {
    driver->partition = static_cast<const char*>(value);
    return true;
  }

  return false;
}


void slurm_driver_init_option_list(stringlist_type * option_list) {
  stringlist_append_copy(option_list, SLURM_PARTITION_OPTION);
  stringlist_append_copy(option_list, SLURM_SBATCH_OPTION);
  stringlist_append_copy(option_list, SLURM_SCONTROL_OPTION);
  stringlist_append_copy(option_list, SLURM_SQUEUE_OPTION);
  stringlist_append_copy(option_list, SLURM_SCANCEL_OPTION);
}


void * slurm_driver_submit_job( void * __driver, const char * cmd, int num_cpu, const char * run_path, const char * job_name, int argc, const char ** argv) {
  slurm_driver_type * driver = slurm_driver_safe_cast( __driver );
  SlurmJob * job = nullptr;
  return job;
}


job_status_type slurm_driver_get_job_status(void * __driver , void * __job) {
  slurm_driver_type * driver = slurm_driver_safe_cast( __driver );
  return JOB_QUEUE_PENDING;
}


void slurm_driver_kill_job(void * __driver , void * __job ) {
  slurm_driver_type * driver = slurm_driver_safe_cast( __driver );
}

void slurm_driver_free_job(void * __job) {
  SlurmJob * job = static_cast<SlurmJob *>(__job);
  delete job;
}
