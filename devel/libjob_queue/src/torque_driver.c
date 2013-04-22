/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'torque_driver.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <string.h>
#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/job_queue/torque_driver.h>


#define TORQUE_DRIVER_TYPE_ID 34873653
#define TORQUE_JOB_TYPE_ID    12312312

struct torque_driver_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * queue_name;
  char * qsub_cmd;
  char * qstat_cmd;
  char * qdel_cmd;
};

struct torque_job_struct {
  UTIL_TYPE_ID_DECLARATION;
  long int torque_jobnr;
  int num_exec_host;
  //char **exec_host;
  char * torque_jobnr_char; /* Used to look up the job status in the qstat cache hash table */
};

UTIL_SAFE_CAST_FUNCTION(torque_driver, TORQUE_DRIVER_TYPE_ID);

static UTIL_SAFE_CAST_FUNCTION_CONST(torque_driver, TORQUE_DRIVER_TYPE_ID)
static UTIL_SAFE_CAST_FUNCTION(torque_job, TORQUE_JOB_TYPE_ID)

void * torque_driver_alloc() {
  torque_driver_type * torque_driver = util_malloc(sizeof * torque_driver);
  UTIL_TYPE_ID_INIT(torque_driver, TORQUE_DRIVER_TYPE_ID);

  torque_driver->queue_name = NULL;
  torque_driver->qsub_cmd = NULL;
  torque_driver->qstat_cmd = NULL;
  torque_driver->qdel_cmd = NULL;

  torque_driver_set_option(torque_driver, TORQUE_QSUB_CMD, TORQUE_DEFAULT_QSUB_CMD);
  torque_driver_set_option(torque_driver, TORQUE_QSTAT_CMD, TORQUE_DEFAULT_QSTAT_CMD);
  torque_driver_set_option(torque_driver, TORQUE_QDEL_CMD, TORQUE_DEFAULT_QDEL_CMD);
  return torque_driver;
}

static void torque_driver_set_qsub_cmd(torque_driver_type * driver, const char * qsub_cmd) {
  driver->qsub_cmd = util_realloc_string_copy(driver->qsub_cmd, qsub_cmd);
}

static void torque_driver_set_qstat_cmd(torque_driver_type * driver, const char * qstat_cmd) {
  driver->qstat_cmd = util_realloc_string_copy(driver->qstat_cmd, qstat_cmd);
}

static void torque_driver_set_qdel_cmd(torque_driver_type * driver, const char * qdel_cmd) {
  driver->qdel_cmd = util_realloc_string_copy(driver->qdel_cmd, qdel_cmd);
}

static void torque_driver_set_queue_name(torque_driver_type * driver, const char * queue_name) {
  driver->queue_name = util_realloc_string_copy(driver->queue_name, queue_name);
}

bool torque_driver_set_option(void * __driver, const char * option_key, const void * value) {
  torque_driver_type * driver = torque_driver_safe_cast(__driver);
  bool has_option = true;
  {
    if (strcmp(TORQUE_QSUB_CMD, option_key) == 0)
      torque_driver_set_qsub_cmd(driver, value);
    else if (strcmp(TORQUE_QSTAT_CMD, option_key) == 0)
      torque_driver_set_qstat_cmd(driver, value);
    else if (strcmp(TORQUE_QDEL_CMD, option_key) == 0)
      torque_driver_set_qdel_cmd(driver, value);
    else if (strcmp(TORQUE_QUEUE, option_key) == 0)
      torque_driver_set_queue_name(driver, value);
    else
      has_option = false;
  }
  return has_option;
}

const void * torque_driver_get_option(const void * __driver, const char * option_key) {
  const torque_driver_type * driver = torque_driver_safe_cast_const(__driver);
  {
    if (strcmp(TORQUE_QSUB_CMD, option_key) == 0)
      return driver->qsub_cmd;
    else if (strcmp(TORQUE_QSTAT_CMD, option_key) == 0)
      return driver->qstat_cmd;
    else if (strcmp(TORQUE_QDEL_CMD, option_key) == 0)
      return driver->qdel_cmd;
    else if (strcmp(TORQUE_QUEUE, option_key) == 0)
      return driver->queue_name;
    else {
      util_abort("%s: option_id:%s not recognized for TORQUE driver \n", __func__, option_key);
      return NULL;
    }
  }
}

torque_job_type * torque_job_alloc() {
  torque_job_type * job;
  job = util_malloc(sizeof * job);
  job->num_exec_host = 0;
  job->torque_jobnr_char = NULL;
  job->torque_jobnr = 0;
  UTIL_TYPE_ID_INIT(job, TORQUE_JOB_TYPE_ID);
  return job;
}

// LSF has num_cpu, but this is not added (yet), not sure what the proper argument to qsub should be.

stringlist_type * torque_driver_alloc_cmd(torque_driver_type * driver,
        const char * torque_stdout,
        const char * job_name,
        const char * submit_cmd,
        int num_cpu,
        int job_argc,
        const char ** job_argv) {


  stringlist_type * argv = stringlist_alloc_new();

  {
    int num_nodes = 1;
    char * resource_string = util_alloc_sprintf("nodes=%d:ppn=%d", num_nodes, num_cpu);
    stringlist_append_ref(argv, "-l");
    stringlist_append_copy(argv, resource_string);
    free(resource_string);
  }

  if (driver->queue_name != NULL) {
    stringlist_append_ref(argv, "-q");
    stringlist_append_ref(argv, driver->queue_name);
  }

  if (job_name != NULL) {
    stringlist_append_ref(argv, "-N");
    stringlist_append_ref(argv, job_name);
  }

  stringlist_append_ref(argv, submit_cmd);
  {
    int iarg;
    for (iarg = 0; iarg < job_argc; iarg++)
      stringlist_append_ref(argv, job_argv[ iarg ]);
  }

  return argv;
}

static int torque_job_parse_qsub_stdout(const torque_driver_type * driver, const char * stdout_file) {
  int jobid;
  {
    FILE * stream = util_fopen(stdout_file, "r");
    char * jobid_string = util_fscanf_alloc_upto(stream, ".", false);

    if (jobid_string == NULL || !util_sscanf_int(jobid_string, &jobid)) {
      char * file_content = util_fread_alloc_file_content(stdout_file, NULL);
      fprintf(stderr, "Failed to get torque job id from file: %s \n", stdout_file);
      fprintf(stderr, "qsub command                      : %s \n", driver->qsub_cmd);
      fprintf(stderr, "File content: [%s]\n", file_content);
      free(file_content);
      util_exit("%s: \n", __func__);
    }
    free(jobid_string);
    fclose(stream);
  }
  return jobid;
}

static int torque_driver_submit_shell_job(torque_driver_type * driver,
        const char * torque_stdout,
        const char * job_name,
        const char * submit_cmd,
        int num_cpu,
        int job_argc,
        const char ** job_argv) {
  int job_id;
  char * tmp_file = util_alloc_tmp_file("/tmp", "enkf-submit", true);
  
  {
    stringlist_type * remote_argv = torque_driver_alloc_cmd(driver, torque_stdout, job_name, submit_cmd, num_cpu, job_argc, job_argv);
    char ** argv = stringlist_alloc_char_ref(remote_argv);
    util_fork_exec(driver->qsub_cmd, stringlist_get_size(remote_argv), (const char **) argv, true, NULL, NULL, NULL, tmp_file, NULL);

    free(argv);
    stringlist_free(remote_argv);
  }
  
  job_id = torque_job_parse_qsub_stdout(driver, tmp_file);

  util_unlink_existing(tmp_file);
  free(tmp_file);

  return job_id;
}

void torque_job_free(torque_job_type * job) {
  util_safe_free(job->torque_jobnr_char);
  free(job);
}

void torque_driver_free_job(void * __job) {
  torque_job_type * job = torque_job_safe_cast(__job);
  torque_job_free(job);
}

void * torque_driver_submit_job(void * __driver,
        const char * submit_cmd,
        int num_cpu,
        const char * run_path,
        const char * job_name,
        int argc,
        const char ** argv) {
  torque_driver_type * driver = torque_driver_safe_cast(__driver);
  torque_job_type * job = torque_job_alloc();
  {
    char * torque_stdout = util_alloc_filename(run_path, job_name, "TORQUE-stdout");
    //pthread_mutex_lock( &driver->submit_lock );

    job->torque_jobnr = torque_driver_submit_shell_job(driver, torque_stdout, job_name, submit_cmd, num_cpu, argc, argv);
    job->torque_jobnr_char = util_alloc_sprintf("%ld", job->torque_jobnr);

    //        hash_insert_ref( driver->my_jobs , job->torque_jobnr_char , NULL );   

    //      pthread_mutex_unlock( &driver->submit_lock );
    free(torque_stdout);
  }

  if (job->torque_jobnr > 0)
    return job;
  else {
    /*
      The submit failed - the queue system shall handle
      NULL return values.
     */
    torque_job_free(job);
    return NULL;
  }
}

static char* torque_driver_get_qstat_status(torque_driver_type * driver, char * jobnr_char) {
  char * status = util_malloc(sizeof (char)*2);
  char * tmp_file = util_alloc_tmp_file("/tmp", "enkf-qstat", true);

  char ** argv = util_calloc(1, sizeof * argv);
  argv[0] = jobnr_char;

  util_fork_exec(driver->qstat_cmd, 1, (const char **) argv, true, NULL, NULL, NULL, tmp_file, NULL);
  FILE *stream = util_fopen(tmp_file, "r");
  bool at_eof = false;
  util_fskip_lines(stream, 2);
  char * line = util_fscanf_alloc_line(stream, &at_eof);
  if (line != NULL) {
    char job_id_full_string[32];
    if (sscanf(line, "%s %*s %*s %*s %s %*s", job_id_full_string, status) == 2) {
      char *dotPtr = strchr(job_id_full_string, '.');
      int dotPosition = dotPtr - job_id_full_string;
      char* job_id_as_char_ptr = util_alloc_substring_copy(job_id_full_string, 0, dotPosition);
      if (strcmp(job_id_as_char_ptr, jobnr_char) != 0) {
        util_abort("%s: Job id input (%d) does not match the one found by qstat (%d)\n", __func__,  jobnr_char, job_id_as_char_ptr);
      }
      free(job_id_as_char_ptr);
    }
    free(line);
  }
  else {
    util_abort("Unable to read qstat's output line number 3 from file: %s", tmp_file);
  }
  
  fclose(stream);
  util_unlink_existing(tmp_file);
  free(tmp_file);

  return status;
}

job_status_type torque_driver_get_job_status(void * __driver, void * __job) {
  torque_driver_type * driver = torque_driver_safe_cast(__driver);
  torque_job_type * job = torque_job_safe_cast(__job);
  char * status = torque_driver_get_qstat_status(driver, job->torque_jobnr_char);
  int result = JOB_QUEUE_FAILED;
  if (strcmp(status, "R") == 0) {
    result = JOB_QUEUE_RUNNING;
  } else if (strcmp(status, "E") == 0) {
    result = JOB_QUEUE_DONE;
  } else if (strcmp(status, "C") == 0) {
    result = JOB_QUEUE_DONE;
  } else if (strcmp(status, "Q") == 0) {
    result = JOB_QUEUE_PENDING;
  }
  else {
    util_abort("Unknown status found (%s), expecting one of R, E, C and Q.\n", status);
  }
  free(status);
  
  return result;
}

void torque_driver_kill_job(void * __driver, void * __job) {
  torque_driver_type * driver = torque_driver_safe_cast(__driver);
  torque_job_type * job = torque_job_safe_cast(__job);
  util_fork_exec(driver->qdel_cmd, 1, (const char **) &job->torque_jobnr_char, true, NULL, NULL, NULL, NULL, NULL);
}

void torque_driver_free(torque_driver_type * driver) {
  util_safe_free(driver->queue_name);
  free(driver->qdel_cmd);
  free(driver->qstat_cmd);
  free(driver->qsub_cmd);

  free(driver);
  driver = NULL;
}

void torque_driver_free__(void * __driver) {
  torque_driver_type * driver = torque_driver_safe_cast(__driver);
  torque_driver_free(driver);
}

