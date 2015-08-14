/*
   Copyright (C) 2015  Statoil ASA, Norway.

   The file 'job_node.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef __JOB_NODE_H__
#define __JOB_NODE_H__


#ifdef __cplusplus
extern "C" {
#endif

#include <ert/job_queue/queue_driver.h>

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

typedef bool (job_callback_ftype)   (void *);
typedef struct job_queue_node_struct job_queue_node_type;


  void job_queue_node_free_error_info( job_queue_node_type * node );
  void job_queue_node_fscanf_EXIT( job_queue_node_type * node );
  void job_queue_node_clear_error_info(job_queue_node_type * node);
  void job_queue_node_clear(job_queue_node_type * node);
  void job_queue_node_free_data(job_queue_node_type * node);
  job_queue_node_type * job_queue_node_alloc( );
  void job_queue_node_free(job_queue_node_type * node);
  job_status_type job_queue_node_get_status(const job_queue_node_type * node);
  void job_queue_node_finalize(job_queue_node_type * node);
  void * job_queue_node_get_data(const job_queue_node_type * node);
  void job_queue_node_free_driver_data( job_queue_node_type * node , queue_driver_type * driver);
  void job_queue_node_driver_kill( job_queue_node_type * node , queue_driver_type * driver);

  void job_queue_node_update_status(job_queue_node_type * node , job_status_type status);

  void job_queue_node_initialize( job_queue_node_type * node , const char * run_path , int num_cpu , const char * job_name , int argc , const char ** argv);
  void job_queue_node_set_ok_file( job_queue_node_type * node , const char * ok_file );
  void job_queue_node_set_exit_file( job_queue_node_type * node , const char * exit_file );
  void job_queue_node_set_cmd( job_queue_node_type * node , const char * run_cmd);
  void job_queue_node_init_callbacks( job_queue_node_type * node ,
                                job_callback_ftype * done_callback,
                                job_callback_ftype * retry_callback,
                                job_callback_ftype * exit_callback,
                                void * callback_arg);

  void job_queue_node_get_wrlock( job_queue_node_type * node);
  void job_queue_node_get_rdlock( job_queue_node_type * node);
  void job_queue_node_unlock( job_queue_node_type * node);
  const char * job_queue_node_get_cmd( const job_queue_node_type * node);
  const char * job_queue_node_get_run_path( const job_queue_node_type * node);
  const char * job_queue_node_get_name( const job_queue_node_type * node);
  const char ** job_queue_node_get_argv( const job_queue_node_type * node);
  int  job_queue_node_get_argc( const job_queue_node_type * node);
  int  job_queue_node_get_num_cpu( const job_queue_node_type * node);
  void job_queue_node_update_data( job_queue_node_type * node , void * data);
  void job_queue_node_inc_submit_attempt( job_queue_node_type * node);
  int  job_queue_node_get_submit_attempt( const job_queue_node_type * node);
  void job_queue_node_reset_submit_attempt( job_queue_node_type * node);
  const char * job_queue_node_get_failed_job( const job_queue_node_type * node);
  const char * job_queue_node_get_error_reason( const job_queue_node_type * node);
  const char * job_queue_node_get_stderr_capture( const job_queue_node_type * node);
  const char * job_queue_node_get_stderr_file( const job_queue_node_type * node);

  time_t job_queue_node_get_sim_start( const job_queue_node_type * node );
  time_t job_queue_node_get_sim_end( const job_queue_node_type * node );
  time_t job_queue_node_get_submit_time( const job_queue_node_type * node );

  const char * job_queue_node_get_ok_file( const job_queue_node_type * node);
  const char * job_queue_node_get_exit_file( const job_queue_node_type * node);

  bool job_queue_node_run_DONE_callback( job_queue_node_type * node );
  bool job_queue_node_run_RETRY_callback( job_queue_node_type * node );
  void job_queue_node_run_EXIT_callback( job_queue_node_type * node );

#ifdef __cplusplus
}
#endif
#endif
