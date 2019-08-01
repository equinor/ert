/*
   Copyright (C) 2018  Equinor ASA, Norway.

   The file 'job_status.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef JOB_STATUS_H
#define JOB_STATUS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/hash.hpp>
/*
                                                                   +---------------------------------+
                                                                   |                                 |
+---------------------------------+                                |  JOB_QUEUE_WAITING              <----------------------------------+
|                                 |                                |                                 <--------------+                   |
|   JOB_QUEUE_NOT_ACTIVE          |                                +---------------+-----------------+              |                   |
|                                 |                                                |                                |                   |
+---------------------------------+                                                |                                |                   |
                                                                                   |                                |                   |
                                                                   +---------------v-----------------+              |                   |
                                                                   |                                 |              |                   |
+---------------------------------+                                |  JOB_QUEUE_SUBMITTED            |              |                   |
|                                 |                                |                                 |              |                   |
|JOB_QUEUE_STATUS_FAILURE         |                                +-------+--------------------+----+              |                   |
|                                 |                                        |                    |                   |                   |
+---------------------------------+                                        |                    |  +----------------+----------------+  |
                                                 +-------------------------v-------+            |  |                                 |  |
                                                 |                                 |            |  | JOB_QUEUE_DO_KILL_NODE_FAILURE  |  |
                                                 | JOB_QUEUE_PENDING               |            |  |                                 |  |
                                                 |                                 |            |  +---------------------^-----------+  |
                                                 +-------------------------+-------+            |                        |              |
                                                                           |                    |                        |              |
                                                                           |    +---------------v-----------------+      |              |
                                                                           |    |                                 |      |              |
                                                                           +----> JOB_QUEUE_RUNNING               +------+              |
                     +----------------------------------------------------------+                                 |                     |
                     |                                                          +---+-------------------+---------+                     |
                     |                                                              |                   |                               |
                     |                                                              |                   |                               |
      +--------------v------------------+    +---------------------------------+    |         +---------v-----------------------+       |
      |                                 |    |                                 |    |         |                                 |       |
      | JOB_QUEUE_DO_KILL               |    | JOB_QUEUE_DONE                  +<---+     +--->  JOB_QUEUE_EXIT                 |       |
      |                                 |    |                                 |          |   |                                 |       |
      +--------------+------------------+    +----------------+----------------+          |   +-----------------+---------------+       |
                     |                                        |                           |                     |                       |
                     |                                        |                           |                     |                       |
                     |                                        |                           |                     |                       |
                     |                       +----------------v----------------+          |   +-----------------v---------------+       |
                     |                       |                                 |          |   |                                 |       |
                     |                       |JOB_QUEUE_RUNNING_DONE_CALLBACK  +----------+   | JOB_QUEUE_RUNNING_EXIT_CALLBACK +-------+
                     |                       |                                 |              |                                 |
                     |                       +----------------+----------------+              +----------------+----------------+
                     |                                        |                                                |
                     |                                        |                                                |
                     |                                        |                                                |
      +--------------v------------------+    +----------------v----------------+              +----------------v----------------+
      |                                 |    |                                 |              |                                 |
      | JOB_QUEUE_IS_KILLED             |    | JOB_QUEUE_SUCCESS               |              |  JOB_QUEUE_FAILED               |
      |                                 |    |                                 |              |                                 |
      +---------------------------------+    +---------------------------------+              +---------------------------------+

*/



/*
  NB: the status count algorithm has a HARD assumption that these
  values are on the 2^N form - without holes in the series.
*/

typedef enum {
    JOB_QUEUE_NOT_ACTIVE            =     1, /* This value is used in external query routines - for jobs which are (currently) not active. */
    JOB_QUEUE_WAITING               =     2, /* A node which is waiting in the internal queue. */
    JOB_QUEUE_SUBMITTED             =     4, /* Internal status: It has has been submitted - the next status update will (should) place it as pending or running. */
    JOB_QUEUE_PENDING               =     8, /* A node which is pending - a status returned by the external system. I.e LSF */
    JOB_QUEUE_RUNNING               =    16, /* The job is running */
    JOB_QUEUE_DONE                  =    32, /* The job is done - but we have not yet checked if the target file is produced */
    JOB_QUEUE_EXIT                  =    64, /* The job has exited - check attempts to determine if we retry or go to complete_fail   */
    JOB_QUEUE_IS_KILLED             =   128, /* The job has been killed, following a  JOB_QUEUE_DO_KILL*/
    JOB_QUEUE_DO_KILL               =   256, /* The the job should be killed, either due to user request, or automated measures - the job can NOT be restarted. */
    JOB_QUEUE_SUCCESS               =   512,
    JOB_QUEUE_RUNNING_DONE_CALLBACK =  1024,
    JOB_QUEUE_RUNNING_EXIT_CALLBACK =  2048,
    JOB_QUEUE_STATUS_FAILURE        =  4096,
    JOB_QUEUE_FAILED                =  8192,
    JOB_QUEUE_DO_KILL_NODE_FAILURE  = 16384,
    JOB_QUEUE_UNKNOWN               = 32768
  } job_status_type;

#define JOB_QUEUE_RUNNING_CALLBACK (JOB_QUEUE_RUNNING_DONE_CALLBACK + JOB_QUEUE_RUNNING_EXIT_CALLBACK)

#define JOB_QUEUE_STATUS_ALL (JOB_QUEUE_NOT_ACTIVE + JOB_QUEUE_WAITING + JOB_QUEUE_SUBMITTED + JOB_QUEUE_PENDING + JOB_QUEUE_RUNNING + JOB_QUEUE_DONE + \
                              JOB_QUEUE_EXIT + JOB_QUEUE_IS_KILLED + JOB_QUEUE_DO_KILL + JOB_QUEUE_SUCCESS + JOB_QUEUE_RUNNING_CALLBACK + \
                              JOB_QUEUE_STATUS_FAILURE + JOB_QUEUE_FAILED + JOB_QUEUE_DO_KILL_NODE_FAILURE + JOB_QUEUE_UNKNOWN) 

#define JOB_QUEUE_MAX_STATE  16

  /*
    All jobs which are in the status set defined by
    JOB_QUEUE_CAN_RESTART can be restarted based on external
    user-input. It is OK to try to restart a job which is not in this
    state - basically nothing should happen.
   */
#define JOB_QUEUE_CAN_RESTART  (JOB_QUEUE_FAILED + JOB_QUEUE_IS_KILLED  +  JOB_QUEUE_SUCCESS)


  /*
    These are the jobs which can be killed. It is OK to try to kill a
    job which is not in this state, the only thing happening is that the
    function job_queue_kill_simulation() wil return false.
   */
#define JOB_QUEUE_CAN_KILL    (JOB_QUEUE_WAITING + JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING + JOB_QUEUE_SUBMITTED + JOB_QUEUE_DO_KILL + JOB_QUEUE_DO_KILL_NODE_FAILURE)

#define JOB_QUEUE_WAITING_STATUS (JOB_QUEUE_WAITING + JOB_QUEUE_PENDING)

#define JOB_QUEUE_CAN_UPDATE_STATUS (JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING + JOB_QUEUE_SUBMITTED + JOB_QUEUE_UNKNOWN)

#define JOB_QUEUE_COMPLETE_STATUS (JOB_QUEUE_IS_KILLED + JOB_QUEUE_SUCCESS + JOB_QUEUE_FAILED)


const char * job_status_get_name(job_status_type status);

#ifdef __cplusplus
}
#endif
#endif
