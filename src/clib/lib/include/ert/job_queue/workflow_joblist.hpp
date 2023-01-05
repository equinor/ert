#ifndef ERT_WORKFLOW_JOBLIST_H
#define ERT_WORKFLOW_JOBLIST_H

#include <ert/job_queue/workflow_job.hpp>

typedef struct workflow_joblist_struct workflow_joblist_type;

extern "C" workflow_joblist_type *workflow_joblist_alloc();
extern "C" void workflow_joblist_free(workflow_joblist_type *joblist);
extern "C" const workflow_job_type *
workflow_joblist_get_job(const workflow_joblist_type *joblist,
                         const char *job_name);
extern "C" void workflow_joblist_add_job(workflow_joblist_type *joblist,
                                         const workflow_job_type *job);
extern "C" bool
workflow_joblist_add_job_from_file(workflow_joblist_type *joblist,
                                   const char *job_name,
                                   const char *config_file);
extern "C" config_parser_type *
workflow_joblist_get_compiler(const workflow_joblist_type *joblist);
extern "C" bool workflow_joblist_has_job(const workflow_joblist_type *joblist,
                                         const char *job_name);
stringlist_type *
workflow_joblist_get_job_names(const workflow_joblist_type *joblist);

#endif
