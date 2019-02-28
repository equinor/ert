/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'queue_driver.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_QUEUE_DRIVER_H
#define ERT_QUEUE_DRIVER_H
#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/hash.hpp>
#include <ert/job_queue/job_status.hpp>

  typedef enum {
    NULL_DRIVER = 0,
    LSF_DRIVER = 1,
    LOCAL_DRIVER = 2,
    RSH_DRIVER = 3,
    TORQUE_DRIVER = 4
  } job_driver_type;

#define JOB_DRIVER_ENUM_SIZE 5

  /*
    The options supported by the base queue_driver.
   */
#define MAX_RUNNING          "MAX_RUNNING"


  typedef struct queue_driver_struct queue_driver_type;

  typedef void * (submit_job_ftype) (void * data, const char * cmd, int num_cpu, const char * run_path, const char * job_name, int argc, const char ** argv);
  typedef void (blacklist_node_ftype) (void *, void *);
  typedef void (kill_job_ftype) (void *, void *);
  typedef job_status_type(get_status_ftype) (void *, void *);
  typedef void (free_job_ftype) (void *);
  typedef void (free_queue_driver_ftype) (void *);
  typedef bool (set_option_ftype) (void *, const char*, const void *);
  typedef const void * (get_option_ftype) (const void *, const char *);
  typedef bool (has_option_ftype) (const void *, const char *);
  typedef void (init_option_list_ftype) (stringlist_type *);


  queue_driver_type * queue_driver_alloc_RSH(const char * rsh_cmd, const hash_type * rsh_hostlist);
  queue_driver_type * queue_driver_alloc_LSF(const char * queue_name, const char * resource_request, const char * remote_lsf_server);
  queue_driver_type * queue_driver_alloc_TORQUE();
  queue_driver_type * queue_driver_alloc_local();
  queue_driver_type * queue_driver_alloc(job_driver_type type);

  void * queue_driver_submit_job(queue_driver_type * driver, const char * run_cmd, int num_cpu, const char * run_path, const char * job_name, int argc, const char ** argv);
  void queue_driver_free_job(queue_driver_type * driver, void * job_data);
  void queue_driver_blacklist_node(queue_driver_type * driver, void * job_data);
  void queue_driver_kill_job(queue_driver_type * driver, void * job_data);
  job_status_type queue_driver_get_status(queue_driver_type * driver, void * job_data);

  const char * queue_driver_get_name(const queue_driver_type * driver);

  bool queue_driver_set_option(queue_driver_type * driver, const char * option_key, const void * value);
  bool queue_driver_unset_option(queue_driver_type * driver, const char * option_key);
  const void * queue_driver_get_option(queue_driver_type * driver, const char * option_key);
  void queue_driver_init_option_list(queue_driver_type * driver, stringlist_type * option_list);

  void queue_driver_free(queue_driver_type * driver);
  void queue_driver_free__(void * driver);

  void queue_driver_set_max_running(queue_driver_type * driver, int max_running);
  int  queue_driver_get_max_running(const queue_driver_type * driver);

  typedef enum {SUBMIT_OK           = 0 ,
                SUBMIT_JOB_FAIL     = 1 , /* Typically no more attempts. */
                SUBMIT_DRIVER_FAIL  = 2 , /* The driver would not take the job - for whatever reason?? */
                SUBMIT_QUEUE_CLOSED = 3 } /* The queue is currently not accepting more jobs - either (temporarilty)
                                             because of pause or it is going down. */   submit_status_type;

  UTIL_IS_INSTANCE_HEADER( queue_driver );


#ifdef __cplusplus
}
#endif
#endif
