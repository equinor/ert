#ifndef ERT_HOOK_WORKFLOW_H
#define ERT_HOOK_WORKFLOW_H

#include <ert/config/config_parser.hpp>

#include <ert/job_queue/workflow.hpp>

typedef enum {
    PRE_SIMULATION = 0,
    POST_SIMULATION = 1,
    PRE_UPDATE = 2,
    POST_UPDATE = 3,
    PRE_FIRST_UPDATE = 4
} hook_run_mode_enum;

typedef struct hook_workflow_struct hook_workflow_type;

hook_workflow_type *hook_workflow_alloc(workflow_type *workflow,
                                        hook_run_mode_enum run_mode);
void hook_workflow_free(hook_workflow_type *hook_workflow);
void hook_workflow_free__(void *arg);

extern "C" workflow_type *
hook_workflow_get_workflow(const hook_workflow_type *hook_workflow);
extern "C" hook_run_mode_enum
hook_workflow_get_run_mode(const hook_workflow_type *hook_workflow);

hook_run_mode_enum hook_workflow_run_mode_from_name(const char *run_mode);

#endif
