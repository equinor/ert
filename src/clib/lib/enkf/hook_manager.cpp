#include <stdlib.h>

#include <ert/res_util/file_utils.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/util/util.h>
#include <ert/util/vector.h>
#include <fmt/printf.h>

#include <ert/config/config_parser.hpp>

#include <ert/job_queue/workflow.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/enkf/site_config.hpp>

#include <algorithm>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

struct hook_manager_struct {
    /** vector of hook_workflow_type instances */
    vector_type *hook_workflow_list;
    ert_workflow_list_type *workflow_list;
    hash_type *input_context;
};

hook_manager_type *
hook_manager_alloc(ert_workflow_list_type *workflow_list,
                   const config_content_type *config_content) {

    hook_manager_type *hook_manager = hook_manager_alloc_default(workflow_list);

    if (config_content)
        hook_manager_init(hook_manager, config_content);

    return hook_manager;
}

void hook_manager_free(hook_manager_type *hook_manager) {
    vector_free(hook_manager->hook_workflow_list);
    hash_free(hook_manager->input_context);
    free(hook_manager);
}

static void hook_manager_add_workflow(hook_manager_type *hook_manager,
                                      const char *workflow_name,
                                      hook_run_mode_enum run_mode) {
    if (ert_workflow_list_has_workflow(hook_manager->workflow_list,
                                       workflow_name)) {
        workflow_type *workflow = ert_workflow_list_get_workflow(
            hook_manager->workflow_list, workflow_name);
        hook_workflow_type *hook = hook_workflow_alloc(workflow, run_mode);
        vector_append_owned_ref(hook_manager->hook_workflow_list, hook,
                                hook_workflow_free__);
    } else {
        fprintf(stderr,
                "** Warning: While hooking workflow: %s not recognized among "
                "the list of loaded workflows.",
                workflow_name);
    }
}

hook_manager_type *
hook_manager_alloc_default(ert_workflow_list_type *workflow_list) {
    hook_manager_type *hook_manager =
        (hook_manager_type *)util_malloc(sizeof *hook_manager);
    hook_manager->workflow_list = workflow_list;

    hook_manager->hook_workflow_list = vector_alloc_new();

    config_parser_type *config = config_alloc();
    config_content_type *site_config_content =
        site_config_alloc_content(config);

    if (config_content_has_item(site_config_content, HOOK_WORKFLOW_KEY)) {
        for (int ihook = 0; ihook < config_content_get_occurences(
                                        site_config_content, HOOK_WORKFLOW_KEY);
             ihook++) {
            const char *workflow_name = config_content_iget(
                site_config_content, HOOK_WORKFLOW_KEY, ihook, 0);
            hook_run_mode_enum run_mode =
                hook_workflow_run_mode_from_name(config_content_iget(
                    site_config_content, HOOK_WORKFLOW_KEY, ihook, 1));
            hook_manager_add_workflow(hook_manager, workflow_name, run_mode);
        }
    }
    config_free(config);
    config_content_free(site_config_content);

    hook_manager->input_context = hash_alloc();

    return hook_manager;
}

hook_manager_type *hook_manager_alloc_full(
    ert_workflow_list_type *workflow_list, const char **hook_workflow_names,
    const char **hook_workflow_run_modes, int hook_workflow_count) {

    hook_manager_type *hook_manager = hook_manager_alloc_default(workflow_list);

    for (int i = 0; i < hook_workflow_count; ++i) {
        const char *workflow_name = hook_workflow_names[i];
        hook_run_mode_enum run_mode =
            hook_workflow_run_mode_from_name(hook_workflow_run_modes[i]);
        hook_manager_add_workflow(hook_manager, workflow_name, run_mode);
    }

    return hook_manager;
}

void hook_manager_init(hook_manager_type *hook_manager,
                       const config_content_type *config_content) {
    if (config_content_has_item(config_content, HOOK_WORKFLOW_KEY)) {
        for (int ihook = 0; ihook < config_content_get_occurences(
                                        config_content, HOOK_WORKFLOW_KEY);
             ihook++) {
            const char *workflow_name = config_content_iget(
                config_content, HOOK_WORKFLOW_KEY, ihook, 0);
            hook_run_mode_enum run_mode =
                hook_workflow_run_mode_from_name(config_content_iget(
                    config_content, HOOK_WORKFLOW_KEY, ihook, 1));
            hook_manager_add_workflow(hook_manager, workflow_name, run_mode);
        }
    }
}

const hook_workflow_type *
hook_manager_iget_hook_workflow(const hook_manager_type *hook_manager,
                                int index) {
    return (hook_workflow_type *)vector_iget(hook_manager->hook_workflow_list,
                                             index);
}

int hook_manager_get_size(const hook_manager_type *hook_manager) {
    return vector_get_size(hook_manager->hook_workflow_list);
}
