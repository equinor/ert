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
