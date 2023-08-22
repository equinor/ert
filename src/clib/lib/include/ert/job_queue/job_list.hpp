#ifndef ERT_JOB_LIST_H
#define ERT_JOB_LIST_H

#include <ert/job_queue/job_node.hpp>
#include <ert/job_queue/queue_driver.hpp>

typedef struct job_list_struct job_list_type;

job_list_type *job_list_alloc();
void job_list_free(job_list_type *job_list);
int job_list_get_size(const job_list_type *job_list);
void job_list_add_job(job_list_type *job_list, job_queue_node_type *job_node);
void job_list_reset(job_list_type *job_list);
void job_list_get_wrlock(job_list_type *list);
void job_list_get_rdlock(job_list_type *list);
void job_list_unlock(job_list_type *list);

#endif
