/*
   Copyright (C) 2015  Equinor ASA, Norway.

   The file 'job_status_test.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <cmath>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>

#include <ert/job_queue/job_queue_status.hpp>
#include <ert/job_queue/queue_driver.hpp>
#include <ert/util/test_util.hpp>


void call_get_status( void * arg ) {
  job_queue_status_type * job_status = job_queue_status_safe_cast( arg );
  job_queue_status_get_count( job_status , pow(2, JOB_QUEUE_MAX_STATE));  // This enum value is completly missing; should give util_abort.
}



void test_create() {
  job_queue_status_type * status = job_queue_status_alloc();
  test_assert_true( job_queue_status_is_instance( status ));
  test_assert_int_equal( job_queue_status_get_count( status , JOB_QUEUE_DONE ) , 0 );
  test_assert_util_abort( "job_queue_status_get_count" , call_get_status , status );
  job_queue_status_free( status );
}


void * add_sim( void * arg ) {
   job_queue_status_type * job_status = job_queue_status_safe_cast( arg );
   job_queue_status_inc( job_status , JOB_QUEUE_WAITING );
   return NULL;
}


void * user_exit( void * arg ) {
   job_queue_status_type * job_status = job_queue_status_safe_cast( arg );
   job_queue_status_transition( job_status , JOB_QUEUE_WAITING  , JOB_QUEUE_DO_KILL);
   return NULL;
}


void * user_done( void * arg ) {
   job_queue_status_type * job_status = job_queue_status_safe_cast( arg );
   job_queue_status_transition( job_status , JOB_QUEUE_WAITING  , JOB_QUEUE_DONE);
   return NULL;
}



void test_update() {
  int N = 15000;
  pthread_t * thread_list = (pthread_t *) util_malloc( 2*N*sizeof * thread_list);
  int num_exit_threads = 0;
  int num_done_threads = 0;
  job_queue_status_type * status = job_queue_status_alloc();

  test_assert_int_equal( 0 , job_queue_status_get_total_count( status ));
  for (int i=0; i < 2*N; i++)
    add_sim( status );
  test_assert_int_equal( 2*N , job_queue_status_get_count( status , JOB_QUEUE_WAITING ));
  test_assert_int_equal( 2*N , job_queue_status_get_count( status , JOB_QUEUE_WAITING + JOB_QUEUE_RUNNING + JOB_QUEUE_DONE));

  {
    int i = 0;
    while (true) {
      int thread_status;

      if ((i % 2) == 0) {
        thread_status = pthread_create( &thread_list[i] , NULL , user_exit , status );
        if (thread_status == 0)
          num_exit_threads++;
        else
          break;
      }  else {
        thread_status = pthread_create( &thread_list[i] , NULL , user_done , status );
        if (thread_status == 0)
          num_done_threads++;
        else
          break;
      }

      i++;
      if (i == N)
        break;
    }
  }
  if ((num_done_threads + num_exit_threads) == 0) {
    fprintf(stderr, "Hmmm - not a single thread created - very suspicious \n");
    exit(1);
  }

  for (int i=0; i < num_done_threads + num_exit_threads; i++)
    pthread_join( thread_list[i] , NULL );

  test_assert_int_equal( 2*N - num_done_threads - num_exit_threads , job_queue_status_get_count( status , JOB_QUEUE_WAITING ));
  test_assert_int_equal( num_exit_threads , job_queue_status_get_count( status , JOB_QUEUE_DO_KILL ));
  test_assert_int_equal( num_done_threads , job_queue_status_get_count( status , JOB_QUEUE_DONE ));

  test_assert_int_equal( num_exit_threads + num_done_threads , job_queue_status_get_count( status , JOB_QUEUE_DO_KILL + JOB_QUEUE_DONE));

  test_assert_int_equal( 2*N , job_queue_status_get_total_count( status ));
  test_assert_int_equal( 2*N , job_queue_status_get_count( status , 2*JOB_QUEUE_DO_KILL_NODE_FAILURE - 1));
  job_queue_status_free( status );
}



/*
  The job_queue_status_inc( ) and the job_queue_status_get_count( )
  functions use two different and independent implementations
  internally; that is the reason for this seemingly quite trivial and
  not-very-interesting test.
*/

void test_index() {
  job_queue_status_type * status = job_queue_status_alloc();

  job_queue_status_inc( status, JOB_QUEUE_NOT_ACTIVE );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_NOT_ACTIVE), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 1);

  job_queue_status_inc( status, JOB_QUEUE_WAITING );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_WAITING), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ),  2);

  job_queue_status_inc( status, JOB_QUEUE_SUBMITTED );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_SUBMITTED), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 3);

  job_queue_status_inc( status, JOB_QUEUE_PENDING );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_PENDING), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 4);

  job_queue_status_inc( status, JOB_QUEUE_RUNNING );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_RUNNING), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 5);

  job_queue_status_inc( status, JOB_QUEUE_DONE );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_DONE), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 6);

  job_queue_status_inc( status, JOB_QUEUE_EXIT );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_EXIT), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 7);

  job_queue_status_inc( status, JOB_QUEUE_IS_KILLED );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_IS_KILLED), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 8);

  job_queue_status_inc( status, JOB_QUEUE_DO_KILL );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_DO_KILL), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 9);

  job_queue_status_inc( status, JOB_QUEUE_SUCCESS );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_SUCCESS), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 10);

  job_queue_status_inc( status, JOB_QUEUE_RUNNING_DONE_CALLBACK );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_RUNNING_DONE_CALLBACK), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 11);

  job_queue_status_inc( status, JOB_QUEUE_RUNNING_EXIT_CALLBACK );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_RUNNING_EXIT_CALLBACK), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 12);

  job_queue_status_inc( status, JOB_QUEUE_STATUS_FAILURE );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_FAILURE), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 13);

  job_queue_status_inc( status, JOB_QUEUE_FAILED );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_FAILED), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 14);

  job_queue_status_inc( status, JOB_QUEUE_DO_KILL_NODE_FAILURE );
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_DO_KILL_NODE_FAILURE), 1);
  test_assert_int_equal( job_queue_status_get_count( status, JOB_QUEUE_STATUS_ALL ), 15);

  job_queue_status_free( status );
}

int main( int argc , char ** argv) {
  util_install_signals();
  test_create();
  test_index();
  test_update();
}
