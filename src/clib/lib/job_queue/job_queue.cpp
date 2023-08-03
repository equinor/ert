#include <chrono>

#include <ert/util/util.hpp>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_queue.hpp>

struct job_queue_struct {
    job_list_type *job_list;
    job_queue_status_type *status;
    /** A pointer to a driver instance (LSF|LOCAL) which actually 'does it'. */
    queue_driver_type *driver;
};

/**
   Observe that if this function is called repeatedly the status might change between
   calls, with the consequence that the total number of jobs does not add up
   properly. Alternatively the function job_queue_export_status_summary(), which does
   proper locking, can be used.
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
    return job_queue_node_kill(node, queue->status, queue->driver);
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
  implementation.
*/
void *job_queue_iget_driver_data(job_queue_type *queue, int job_index) {
    void *driver_data;
    ASSIGN_LOCKED_ATTRIBUTE(driver_data, job_queue_node_get_driver_data, node);
    return driver_data;
}

/**
   Observe that the job_queue returned by this function is NOT ready
   for use; a driver must be set explicitly with a call to
   job_queue_set_driver() first.
*/
job_queue_type *job_queue_alloc() {
    job_queue_type *queue = (job_queue_type *)util_malloc(sizeof *queue);
    queue->driver = NULL;
    queue->job_list = job_list_alloc();
    queue->status = job_queue_status_alloc();
    return queue;
}

void job_queue_set_driver(job_queue_type *queue, queue_driver_type *driver) {
    queue->driver = driver;
}

void job_queue_free(job_queue_type *queue) {
    job_list_free(queue->job_list);
    job_queue_status_free(queue->status);
    free(queue);
}

int job_queue_add_job_node(job_queue_type *queue, job_queue_node_type *node) {
    job_list_get_wrlock(queue->job_list);
    job_list_add_job(queue->job_list, node);
    job_queue_node_status_transition(node, queue->status, JOB_QUEUE_WAITING);
    int queue_index = job_queue_node_get_queue_index(node);
    job_list_unlock(queue->job_list);
    return queue_index;
}
