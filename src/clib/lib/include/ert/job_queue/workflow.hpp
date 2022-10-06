#ifndef ERT_WORKFLOW_H
#define ERT_WORKFLOW_H

#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_error.hpp>

#include <ert/job_queue/workflow_job.hpp>
#include <ert/job_queue/workflow_joblist.hpp>

typedef struct workflow_struct workflow_type;

extern "C" const config_error_type *
workflow_get_last_error(const workflow_type *workflow);
extern "C" workflow_type *workflow_alloc(const char *src_file,
                                         workflow_joblist_type *joblist);
bool workflow_run(workflow_type *workflow, void *self, bool verbose,
                  const subst_list_type *context);
extern "C" void workflow_free(workflow_type *workflow);
void workflow_free__(void *arg);

int workflow_get_stack_size(const workflow_type *workflow);
void *workflow_iget_stack_ptr(const workflow_type *workflow, int index);
void *workflow_pop_stack(workflow_type *workflow);

extern "C" int workflow_size(const workflow_type *workflow);
extern "C" const workflow_job_type *
workflow_iget_job(const workflow_type *workflow, int index);
extern "C" stringlist_type *
workflow_iget_arguments(const workflow_type *workflow, int index);
extern "C" bool workflow_try_compile(workflow_type *script,
                                     const subst_list_type *context);

UTIL_IS_INSTANCE_HEADER(workflow);

#endif
