#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ert/util/util.hpp>

#include <ert/job_queue/job_list.hpp>
#include <ert/job_queue/job_node.hpp>

struct job_list_struct {
    int active_size;
    int alloc_size;
    job_queue_node_type **jobs;
    pthread_rwlock_t lock;
};

job_list_type *job_list_alloc() {
    job_list_type *job_list = (job_list_type *)util_malloc(sizeof *job_list);
    job_list->active_size = 0;
    job_list->alloc_size = 0;
    job_list->jobs = NULL;
    pthread_rwlock_init(&job_list->lock, NULL);
    return job_list;
}

void job_list_reset(job_list_type *job_list) {
    int queue_index;
    for (queue_index = 0; queue_index < job_list->active_size; queue_index++) {
        job_queue_node_type *node = job_list->jobs[queue_index];
        job_queue_node_free(node);
        job_list->jobs[queue_index] = NULL;
    }
    job_list->active_size = 0;
}

int job_list_get_size(const job_list_type *job_list) {
    return job_list->active_size;
}

/**
  This takes ownership to the job node instance.
*/
void job_list_add_job(job_list_type *job_list, job_queue_node_type *job_node) {
    if (job_list->alloc_size == job_list->active_size) {
        int new_alloc_size = util_int_max(16, job_list->alloc_size * 2);
        job_list->jobs = (job_queue_node_type **)util_realloc(
            job_list->jobs, sizeof *job_list->jobs * new_alloc_size);
        job_list->alloc_size = new_alloc_size;
    }

    {
        int queue_index = job_list_get_size(job_list);
        job_queue_node_set_queue_index(job_node, queue_index);
        job_list->jobs[queue_index] = job_node;
    }
    job_list->active_size++;
}

void job_list_free(job_list_type *job_list) {
    if (job_list->alloc_size > 0) {
        job_list_reset(job_list);
        free(job_list->jobs);
    }
    free(job_list);
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
