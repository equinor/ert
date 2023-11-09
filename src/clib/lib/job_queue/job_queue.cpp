#include <chrono>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_queue.hpp>

struct job_queue_struct {
    job_list_type *job_list = nullptr;
    /** A pointer to a driver instance (LSF|LOCAL) which actually 'does it'. */
    queue_driver_type *driver = nullptr;
};

job_queue_type *job_queue_alloc(queue_driver_type *driver) {
    auto queue = new job_queue_type;
    queue->job_list = job_list_alloc();
    queue->driver = driver;
    return queue;
}

void job_queue_free(job_queue_type *queue) {
    job_list_free(queue->job_list);
    delete queue;
}

int job_queue_add_job_node(job_queue_type *queue, job_queue_node_type *node) {
    job_list_get_wrlock(queue->job_list);
    job_list_add_job(queue->job_list, node);

    pthread_mutex_lock(&node->data_mutex);

    if (job_queue_node_get_status(node) != JOB_QUEUE_WAITING)
        job_queue_node_set_status(node, JOB_QUEUE_WAITING);

    pthread_mutex_unlock(&node->data_mutex);

    int queue_index = job_queue_node_get_queue_index(node);
    job_list_unlock(queue->job_list);
    return queue_index;
}
