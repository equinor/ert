#include <stdlib.h>
#include <string.h>

#include <ert/util/type_macros.h>
#include <ert/util/util.h>

#include <ert/job_queue/workflow.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/hook_workflow.hpp>

#define RUN_MODE_PRE_FIRST_UPDATE_NAME "PRE_FIRST_UPDATE"
#define RUN_MODE_PRE_SIMULATION_NAME "PRE_SIMULATION"
#define RUN_MODE_POST_SIMULATION_NAME "POST_SIMULATION"
#define RUN_MODE_PRE_UPDATE_NAME "PRE_UPDATE"
#define RUN_MODE_POST_UPDATE_NAME "POST_UPDATE"

#define HOOK_WORKFLOW_TYPE_ID 7321780

struct hook_workflow_struct {
    UTIL_TYPE_ID_DECLARATION;
    hook_run_mode_enum run_mode;
    workflow_type *workflow;
};

static UTIL_SAFE_CAST_FUNCTION(hook_workflow, HOOK_WORKFLOW_TYPE_ID);

hook_workflow_type *hook_workflow_alloc(workflow_type *workflow,
                                        hook_run_mode_enum run_mode) {
    hook_workflow_type *hook_workflow =
        (hook_workflow_type *)util_malloc(sizeof *hook_workflow);
    UTIL_TYPE_ID_INIT(hook_workflow, HOOK_WORKFLOW_TYPE_ID);
    hook_workflow->run_mode = run_mode;
    hook_workflow->workflow = workflow;
    return hook_workflow;
}

void hook_workflow_free(hook_workflow_type *hook_workflow) {
    free(hook_workflow);
}

void hook_workflow_free__(void *arg) {
    hook_workflow_type *hook_workflow = hook_workflow_safe_cast(arg);
    hook_workflow_free(hook_workflow);
}

workflow_type *
hook_workflow_get_workflow(const hook_workflow_type *hook_workflow) {
    return hook_workflow->workflow;
}

hook_run_mode_enum hook_workflow_run_mode_from_name(const char *run_mode) {
    hook_run_mode_enum mode;
    if (strcmp(run_mode, RUN_MODE_PRE_SIMULATION_NAME) == 0)
        mode = PRE_SIMULATION;
    else if (strcmp(run_mode, RUN_MODE_POST_SIMULATION_NAME) == 0)
        mode = POST_SIMULATION;
    else if (strcmp(run_mode, RUN_MODE_PRE_UPDATE_NAME) == 0)
        mode = PRE_UPDATE;
    else if (strcmp(run_mode, RUN_MODE_POST_UPDATE_NAME) == 0)
        mode = POST_UPDATE;
    else if (strcmp(run_mode, RUN_MODE_PRE_FIRST_UPDATE_NAME) == 0)
        mode = PRE_FIRST_UPDATE;
    else {
        util_abort("%s: unrecognized run mode :%s \n", __func__, run_mode);
        mode = POST_UPDATE; /* Dummy */
    }
    return mode;
}

hook_run_mode_enum
hook_workflow_get_run_mode(const hook_workflow_type *hook_workflow) {
    return hook_workflow->run_mode;
}
