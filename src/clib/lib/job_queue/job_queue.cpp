#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ert/logging.hpp>
#include <ert/util/util.hpp>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_queue.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("job_queue");

/*

   The running of external jobs is handled through an abstract
   job_queue implemented in this file; the job_queue then contains a
   'driver' which actually runs the job. All drivers must support the
   following functions

     submit: This will submit a job, and return a pointer to a
             newly allocated queue_job instance.

     clean:  This will clear up all resources used by the job.

     abort:  This will stop the job, and then call clean.

     status: This will get the status of the job.


   When calling the various driver functions the queue layer needs to
   dereference the driver structures, i.e. to get access to the
   driver->submit_jobs function. This is currently (rather clumsily??
   implemented like this):

        When implementing a driver the driver struct MUST start like
        this:

        struct some_driver {
            QUEUE_DRIVER_FUNCTIONS
            ....
            ....
        }

        The function allocating a driver instance will just return a
        (void *) however in the queue layer the driver is stored as a
        queue_driver_type instance which is a struct like this:

        struct queue_driver_struct {
            QUEUE_DRIVER_FIELDS
        }

        I.e. it only contains the pointers common to all the driver
        implementations. When calling a driver function the spesific
        driver will cast to it's datatype.

   Observe that this library also contains the files ext_joblist and
   ext_job, those files implement a particular way of dispatching
   external jobs in a series; AFTER THEY HAVE BEEN SUBMITTED. So seen
   from the this scope those files do not provide any particluar
   functionality; there is no compile-time dependencies either.
*/

/*
  Some words about status
  =======================

  The status of a particular job is given by the job_status field of
  the job_queue_node_type, the possible values are given by the enum
  job_status_type, defined in queue_driver.h.

  To actually __GET__ the status of a job we use the driver->status()
  function which will invoke a driver specific function and return the
  new status.

    1. The driver->status() function is invoked by the
       job_queue_update_status() function. This should be invoked by
       the same thread as is running the main queue management in
       job_queue_run_jobs().


    2. The actual change of status is handled by the function
       job_queue_change_node_status(); arbitrary assignments of the
       type job->status = new_status is STRICTLY ILLEGAL.


    3. When external functions query about the status of a particular
       job they get the status value currently stored (i.e. cached) in
       the job_node; external scope can NOT initiate a
       driver->status() function call.

       This might result in external scope getting a outdated status -
       live with it.


    4. The name 'status' indicates that this is read-only property;
       that is actually not the case. In the main manager function
       job_queue_run_jobs() action is taken based on the value of the
       status field, and to initiate certain action on jobs the queue
       system (and also external scope) can explicitly set the status
       of a job (by using the job_queue_change_node_status() function).

       The most prominent example of this is when we want to run a
       certain job again, that is achieved with:

           job_queue_node_change_status( queue , node , JOB_QUEUE_WAITING );

       When the queue manager subsequently finds the job with status
       'JOB_QUEUE_WAITING' it will (re)submit this job.
*/

#define JOB_QUEUE_TYPE_ID 665210

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
struct job_queue_struct {
    job_list_type *job_list;
    job_queue_status_type *status;
    /** The queue will look for the occurrence of this file to detect a failure. */
    char *exit_file;
    /** The queue will look for this file to verify that the job is running or
     * has run.  If not, ok_file is ignored. */
    char *status_file;
    /** A pointer to a driver instance (LSF|LOCAL) which actually 'does it'. */
    queue_driver_type *driver;

    /** True if the queue has been reset and is ready for use, false if the
     * queue has been used and not reset */
    bool open;
    /** If there comes an external signal to abandon the whole thing user_exit
     * will be set to true, and things start to dwindle down. */
    bool user_exit;
    bool running;
    bool pause_on;

    /** The maximum number of submit attempts for one job. */
    int max_submit;
    /** Maximum allowed time for a job to run, 0 = unlimited */
    int max_duration;
    /** A job is only allowed to run until this time. 0 = no time set, ignore stop_time */
    time_t progress_timestamp;
    /** The sleep time before checking for updates. */
    unsigned long usleep_time;
    /** This mutex is used to ensure that ONLY one thread is executing the job_queue_run_jobs(). */
    std::mutex run_mutex;

    /** This holds future results of currently running callbacks */
    std::vector<std::future<void>> active_callbacks;
};

/*
  Must hold on to:

   1. A write lock for the job node.
   3. A read lock for the job_list

*/
static bool job_queue_change_node_status(job_queue_type *queue,
                                         job_queue_node_type *node,
                                         job_status_type new_status) {
    return job_queue_node_status_transition(node, queue->status, new_status);
}

/**
   Will return the number of jobs with status @status.

      #include <queue_driver.h>

      printf("Running jobs...: %03d \n", job_queue_iget_status_summary( queue , JOB_QUEUE_RUNNING ));
      printf("Waiting jobs:..: %03d \n", job_queue_iget_status_summary( queue , JOB_QUEUE_WAITING ));

   Observe that if this function is called repeatedly the status might change between
   calls, with the consequence that the total number of jobs does not add up
   properly. The handles itself autonomously so as long as the return value from this
   function is only used for information purposes this does not matter. Alternatively
   the function job_queue_export_status_summary(), which does proper locking, can be
   used.
*/
int job_queue_iget_status_summary(const job_queue_type *queue,
                                  job_status_type status) {
    return job_queue_status_get_count(queue->status, status);
}

int job_queue_get_num_running(const job_queue_type *queue) {
    return job_queue_iget_status_summary(queue, JOB_QUEUE_RUNNING);
}

int job_queue_get_num_pending(const job_queue_type *queue) {
    return job_queue_iget_status_summary(queue, JOB_QUEUE_PENDING);
}

int job_queue_get_num_waiting(const job_queue_type *queue) {
    return job_queue_iget_status_summary(queue, JOB_QUEUE_WAITING);
}

int job_queue_get_num_complete(const job_queue_type *queue) {
    return job_queue_iget_status_summary(queue, JOB_QUEUE_SUCCESS);
}

void job_queue_set_max_job_duration(job_queue_type *queue,
                                    int max_duration_seconds) {
    queue->max_duration = max_duration_seconds;
}

/**
   Observe that jobs with status JOB_QUEUE_WAITING can also be killed; for those
   jobs the kill should be interpreted as "Forget about this job for now and set
   the status JOB_QUEUE_IS_KILLED", however it is important that we not call
   the driver->kill() function on it because the job slot will have no data
   (i.e. LSF jobnr), and the driver->kill() function will fail if presented with
   such a job.

   Only jobs which have a status matching "JOB_QUEUE_CAN_KILL" can be
   killed; if the job is not in a killable state the function will do
   nothing. This includes trying to kill a job which is not even
   found.

   Observe that jobs (slots) with status JOB_QUEUE_NOT_ACTIVE can NOT be
   meaningfully killed; that is because these jobs have not yet been submitted
   to the queue system, and there is not yet established a mapping between
   external id and queue_index.

   Must hold on to joblist:read lock.
*/
static bool job_queue_kill_job_node(job_queue_type *queue,
                                    job_queue_node_type *node) {
    bool result = job_queue_node_kill(node, queue->status, queue->driver);
    return result;
}

class JobListReadLock {
    /* This is just a trick to make sure list is unlocked when exiting scope,
 * also when exiting due to exceptions */
public:
    JobListReadLock(job_list_type *job_list) : job_list(job_list) {
        job_list_get_rdlock(this->job_list);
    }
    ~JobListReadLock() { job_list_unlock(this->job_list); }

private:
    job_list_type *job_list;
};

#define ASSIGN_LOCKED_ATTRIBUTE(var, func, ...)                                \
    {                                                                          \
        JobListReadLock rl(queue->job_list);                                   \
        job_queue_node_type *node =                                            \
            job_list_iget_job(queue->job_list, job_index);                     \
        var = func(__VA_ARGS__);                                               \
    }

bool job_queue_kill_job(job_queue_type *queue, int job_index) {
    bool result;
    ASSIGN_LOCKED_ATTRIBUTE(result, job_queue_kill_job_node, queue, node);
    return result;
}

/**
  This returns a pointer to a very internal datastructure; used by the
  Job class in Python which interacts directly with the driver
  implementation. This is too low level, and the whole Driver / Job
  implementation in Python should be changed to only expose the higher
  level queue class.
*/
void *job_queue_iget_driver_data(job_queue_type *queue, int job_index) {
    void *driver_data;
    ASSIGN_LOCKED_ATTRIBUTE(driver_data, job_queue_node_get_driver_data, node);
    return driver_data;
}

bool job_queue_is_running(const job_queue_type *queue) {
    return queue->running;
}

bool job_queue_accept_jobs(const job_queue_type *queue) {
    if (queue->user_exit)
        return false;

    return queue->open;
}

/**
   Observe that the job_queue returned by this function is NOT ready
   for use; a driver must be set explicitly with a call to
   job_queue_set_driver() first.
*/
job_queue_type *job_queue_alloc(int max_submit, const char *status_file,
                                const char *exit_file) {

    job_queue_type *queue = (job_queue_type *)util_malloc(sizeof *queue);
    queue->usleep_time = 250000; /* 1000000 : 1 second */
    queue->max_duration = 0;
    queue->max_submit = max_submit;
    queue->driver = NULL;
    queue->exit_file = util_alloc_string_copy(exit_file);
    queue->status_file = util_alloc_string_copy(status_file);
    queue->open = true;
    queue->user_exit = false;
    queue->pause_on = false;
    queue->running = false;
    queue->job_list = job_list_alloc();
    queue->status = job_queue_status_alloc();
    queue->progress_timestamp = time(NULL);

    return queue;
}

/**
   The calling scope must retain a handle to the current driver and
   free it.  Should (in principle) be possible to change driver on a
   running system whoaaa. Will read and update the max_running value
   from the driver.
*/
void job_queue_set_driver(job_queue_type *queue, queue_driver_type *driver) {
    queue->driver = driver;
}

int job_queue_get_max_submit(const job_queue_type *job_queue) {
    return job_queue->max_submit;
}

void job_queue_free(job_queue_type *queue) {
    free(queue->exit_file);
    free(queue->status_file);
    job_list_free(queue->job_list);
    job_queue_status_free(queue->status);
    free(queue);
}

char *job_queue_get_exit_file(const job_queue_type *queue) {
    return queue->exit_file;
}

char *job_queue_get_status_file(const job_queue_type *queue) {
    return queue->status_file;
}

int job_queue_add_job_node(job_queue_type *queue, job_queue_node_type *node) {
    job_list_get_wrlock(queue->job_list);

    job_list_add_job(queue->job_list, node);
    job_queue_change_node_status(queue, node, JOB_QUEUE_WAITING);
    int queue_index = job_queue_node_get_queue_index(node);
    job_list_unlock(queue->job_list);
    return queue_index;
}
