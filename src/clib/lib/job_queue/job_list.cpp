#include <pthread.h>
#include <vector>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_node.hpp>

struct job_list_struct {
    std::vector<job_queue_node_type *> vec_jobs;
    pthread_rwlock_t lock{};
};

job_list_type *job_list_alloc() { return new job_list_type; }

void job_list_add_job(job_list_type *job_list, job_queue_node_type *job_node) {
    unsigned long queue_index = job_list->vec_jobs.size();
    job_queue_node_set_queue_index(job_node, static_cast<int>(queue_index));
    job_list->vec_jobs.push_back(job_node);
}

size_t job_list_get_size(const job_list_type *job_list) {
    return job_list->vec_jobs.size();
}

void job_list_free(job_list_type *job_list) {
    for (auto &vec_job : job_list->vec_jobs)
        job_queue_node_free(vec_job);
    job_list->vec_jobs.clear();
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
