#include <pthread.h>
#include <vector>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_node.hpp>

struct job_list_struct {
    std::vector<job_queue_node_type *> vec_jobs;
    pthread_rwlock_t lock;
};

job_list_type *job_list_alloc() {
    auto job_list = new job_list_type;
    pthread_rwlock_init(&job_list->lock, nullptr);
    return job_list;
}

void job_list_reset(job_list_type *job_list) {
    for (auto &vec_job : job_list->vec_jobs)
        job_queue_node_free(vec_job);
    job_list->vec_jobs.clear();
}

int job_list_get_size(const job_list_type *job_list) {
    return job_list->vec_jobs.size();
}

void job_list_add_job(job_list_type *job_list, job_queue_node_type *job_node) {
    int queue_index = job_list->vec_jobs.size();
    job_queue_node_set_queue_index(job_node, queue_index);
    job_list->vec_jobs.push_back(job_node);
}

void job_list_free(job_list_type *job_list) {
    job_list_reset(job_list);
    delete job_list;
}

void job_list_get_wrlock(job_list_type *list) {
    pthread_rwlock_wrlock(&list->lock);
}

void job_list_get_rdlock(job_list_type *list) {
    pthread_rwlock_rdlock(&list->lock);
}

void job_list_unlock(job_list_type *list) {
    pthread_rwlock_unlock(&list->lock);
}
