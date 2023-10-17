#ifndef JOB_STATUS_H
#define JOB_STATUS_H

#include <map>
#include <string>

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
                     |                                        |                           |                     ------------------------+
                     |                                        |                           |                     |                        
                     |                                        ----------------------------+                     |                        
                     |                                        |                                                 |                        
                     |                                        |                                                 |                        
                     |                                        |                                                 |
                     |                                        |                                                 |
                     |                                        |                                                 |
      +--------------v------------------+    +----------------v----------------+              +-----------------v---------------+
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
    /** This value is used in external query routines - for jobs which are
     * (currently) not active. */
    JOB_QUEUE_NOT_ACTIVE = 1,
    /** A node which is waiting in the internal queue. */
    JOB_QUEUE_WAITING = 2,
    /** Internal status: It has has been submitted - the next status update
     * will (should) place it as pending or running. */
    JOB_QUEUE_SUBMITTED = 4,
    /** A node which is pending - a status returned by the external system. I.e LSF */
    JOB_QUEUE_PENDING = 8,
    /** The job is running */
    JOB_QUEUE_RUNNING = 16,
    /** The job is done - but we have not yet checked if the target file is produced */
    JOB_QUEUE_DONE = 32,
    /** The job has exited - check attempts to determine if we retry or go to complete_fail*/
    JOB_QUEUE_EXIT = 64,
    /** The job has been killed, following a JOB_QUEUE_DO_KILL*/
    JOB_QUEUE_IS_KILLED = 128,
    /** The job should be killed, either due to user request, or automated
     * measures - the job can NOT be restarted. */
    JOB_QUEUE_DO_KILL = 256,
    JOB_QUEUE_SUCCESS = 512,
    JOB_QUEUE_STATUS_FAILURE = 1024,
    JOB_QUEUE_FAILED = 2048,
    JOB_QUEUE_DO_KILL_NODE_FAILURE = 4096,
    JOB_QUEUE_UNKNOWN = 8192,
} job_status_type;

const std::map<const job_status_type, const std::string> job_status_names = {
    {JOB_QUEUE_NOT_ACTIVE, "JOB_QUEUE_NOT_ACTIVE"},
    {JOB_QUEUE_WAITING, "JOB_QUEUE_WAITING"},
    {JOB_QUEUE_SUBMITTED, "JOB_QUEUE_SUBMITTED"},
    {JOB_QUEUE_PENDING, "JOB_QUEUE_PENDING"},
    {JOB_QUEUE_RUNNING, "JOB_QUEUE_RUNNING"},
    {JOB_QUEUE_DONE, "JOB_QUEUE_DONE"},
    {JOB_QUEUE_EXIT, "JOB_QUEUE_EXIT"},
    {JOB_QUEUE_IS_KILLED, "JOB_QUEUE_IS_KILLED"},
    {JOB_QUEUE_DO_KILL, "JOB_QUEUE_DO_KILL"},
    {JOB_QUEUE_SUCCESS, "JOB_QUEUE_SUCCESS"},
    {JOB_QUEUE_STATUS_FAILURE, "JOB_QUEUE_STATUS_FAILURE"},
    {JOB_QUEUE_FAILED, "JOB_QUEUE_FAILED"},
    {JOB_QUEUE_DO_KILL_NODE_FAILURE, "JOB_QUEUE_DO_KILL_NODE_FAILURE"},
    {JOB_QUEUE_UNKNOWN, "JOB_QUEUE_UNKNOWN"},
};

#define JOB_QUEUE_MAX_STATE 14
#define JOB_QUEUE_STATUS_ALL ((1 << JOB_QUEUE_MAX_STATE) - 1)

/*
    These are the jobs which can be killed. It is OK to try to kill a
    job which is not in this state, the only thing happening is that the
    function job_queue_kill_simulation() wil return false.
   */
#define JOB_QUEUE_CAN_KILL                                                     \
    (JOB_QUEUE_WAITING + JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING +               \
     JOB_QUEUE_SUBMITTED + JOB_QUEUE_DO_KILL + JOB_QUEUE_DO_KILL_NODE_FAILURE)

#define JOB_QUEUE_CAN_UPDATE_STATUS                                            \
    (JOB_QUEUE_RUNNING + JOB_QUEUE_PENDING + JOB_QUEUE_SUBMITTED +             \
     JOB_QUEUE_UNKNOWN)

#define JOB_QUEUE_COMPLETE_STATUS                                              \
    (JOB_QUEUE_IS_KILLED + JOB_QUEUE_SUCCESS + JOB_QUEUE_FAILED)

#endif
