#ifndef ERT_HOOK_MANAGER_H
#define ERT_HOOK_MANAGER_H

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/hook_workflow.hpp>
#include <vector>

typedef struct hook_manager_struct hook_manager_type;

hook_manager_type *
hook_manager_alloc_default(ert_workflow_list_type *workflow_list);
extern "C" hook_manager_type *hook_manager_alloc(ert_workflow_list_type *,
                                                 const config_content_type *);

extern "C" PY_USED hook_manager_type *hook_manager_alloc_full(
    ert_workflow_list_type *workflow_list, const char **hook_workflow_names,
    const char **hook_workflow_run_modes, int hook_workflow_count);

extern "C" void hook_manager_free(hook_manager_type *hook_manager);

void hook_manager_init(hook_manager_type *hook_manager,
                       const config_content_type *config);
void hook_manager_run_workflows(const hook_manager_type *hook_manager,
                                hook_run_mode_enum run_mode, void *self);

extern "C" PY_USED const hook_workflow_type *
hook_manager_iget_hook_workflow(const hook_manager_type *hook_manager,
                                int index);
extern "C" int hook_manager_get_size(const hook_manager_type *hook_manager);
#endif
