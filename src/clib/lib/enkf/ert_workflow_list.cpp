#include <filesystem>
#include <set>
#include <string>

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/hash.h>
#include <ert/util/vector.h>

#include <ert/config/config_parser.hpp>
#include <ert/enkf/hook_workflow.hpp>

#include <ert/job_queue/workflow.hpp>

#include <ert/logging.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/model_config.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("enkf");

struct ert_workflow_list_struct {
    hash_type *workflows;
    hash_type *alias_map;
    workflow_joblist_type *joblist;
    vector_type *hook_workflow_list;
    const subst_list_type *context;
    const config_error_type *last_error;
    bool verbose;
};

static void ert_workflow_list_init(ert_workflow_list_type *workflow_list,
                                   const config_content_type *config);

ert_workflow_list_type *
ert_workflow_list_alloc_empty(const subst_list_type *context) {
    ert_workflow_list_type *workflow_list =
        (ert_workflow_list_type *)util_malloc(sizeof *workflow_list);
    workflow_list->workflows = hash_alloc();
    workflow_list->alias_map = hash_alloc();
    workflow_list->joblist = workflow_joblist_alloc();
    workflow_list->context = context;
    workflow_list->last_error = NULL;
    workflow_list->hook_workflow_list = vector_alloc_new();
    ert_workflow_list_set_verbose(workflow_list, DEFAULT_WORKFLOW_VERBOSE);
    return workflow_list;
}

ert_workflow_list_type *
ert_workflow_list_alloc(const subst_list_type *context,
                        const config_content_type *config_content,
                        const config_content_type *site_config_content) {

    ert_workflow_list_type *workflow_list =
        ert_workflow_list_alloc_empty(context);
    ert_workflow_list_init(workflow_list, site_config_content);

    if (config_content)
        ert_workflow_list_init(workflow_list, config_content);

    return workflow_list;
}

ert_workflow_list_type *ert_workflow_list_alloc_full(
    const subst_list_type *context, workflow_joblist_type *workflow_joblist,
    const char **hook_workflow_names, const char **hook_workflow_run_modes,
    int hook_workflow_count) {
    ert_workflow_list_type *workflow_list =
        ert_workflow_list_alloc_empty(context);
    workflow_list->joblist = workflow_joblist;
    workflow_list->context = context;
    for (int i = 0; i < hook_workflow_count; ++i) {
        const char *workflow_name = hook_workflow_names[i];
        hook_run_mode_enum run_mode =
            hook_workflow_run_mode_from_name(hook_workflow_run_modes[i]);
        workflow_type *workflow =
            ert_workflow_list_get_workflow(workflow_list, workflow_name);
        hook_workflow_type *hook = hook_workflow_alloc(workflow, run_mode);
        vector_append_owned_ref(workflow_list->hook_workflow_list, hook,
                                hook_workflow_free__);
    }

    return workflow_list;
}

void ert_workflow_list_set_verbose(ert_workflow_list_type *workflow_list,
                                   bool verbose) {
    workflow_list->verbose = verbose;
}

const subst_list_type *
ert_workflow_list_get_context(const ert_workflow_list_type *workflow_list) {
    return workflow_list->context;
}

void ert_workflow_list_free(ert_workflow_list_type *workflow_list) {
    hash_free(workflow_list->workflows);
    hash_free(workflow_list->alias_map);
    vector_free(workflow_list->hook_workflow_list);
    workflow_joblist_free(workflow_list->joblist);
    free(workflow_list);
}

workflow_type *
ert_workflow_list_add_workflow(ert_workflow_list_type *workflow_list,
                               const char *workflow_file,
                               const char *workflow_name) {
    if (fs::exists(workflow_file)) {
        workflow_type *workflow =
            workflow_alloc(workflow_file, workflow_list->joblist);
        char *name;

        if (workflow_name == NULL)
            util_alloc_file_components(workflow_file, NULL, &name, NULL);
        else
            name = (char *)workflow_name;

        hash_insert_hash_owned_ref(workflow_list->workflows, name, workflow,
                                   workflow_free__);
        if (hash_has_key(workflow_list->alias_map, name))
            hash_del(workflow_list->alias_map, name);

        if (workflow_name == NULL)
            free(name);

        return workflow;
    } else
        return NULL;
}

void ert_workflow_list_add_alias(ert_workflow_list_type *workflow_list,
                                 const char *real_name, const char *alias) {
    if (!util_string_equal(real_name, alias))
        hash_insert_ref(workflow_list->alias_map, alias, real_name);
}

void ert_workflow_list_add_job(ert_workflow_list_type *workflow_list,
                               const char *job_name, const char *config_file) {
    char *name = (char *)job_name;

    if (job_name == NULL)
        util_alloc_file_components(config_file, NULL, &name, NULL);

    if (!workflow_joblist_add_job_from_file(workflow_list->joblist, name,
                                            config_file))
        fprintf(stderr, "** Warning: failed to add workflow job:%s from:%s \n",
                name, config_file);

    if (job_name == NULL)
        free(name);
}

bool ert_workflow_list_has_job(const ert_workflow_list_type *workflow_list,
                               const char *job_name) {
    return workflow_joblist_has_job(workflow_list->joblist, job_name);
}

const workflow_job_type *
ert_workflow_list_get_job(const ert_workflow_list_type *workflow_list,
                          const char *job_name) {
    return workflow_joblist_get_job(workflow_list->joblist, job_name);
}

static char *ert_workflow_list_alloc_name(const char *path,
                                          const char *root_name) {
    char *full_path =
        util_alloc_sprintf("%s%s%s", path, UTIL_PATH_SEP_STRING, root_name);

    if (util_is_file(full_path))
        return full_path;
    else
        free(full_path);

    return NULL;
}

void ert_workflow_list_add_jobs_in_directory(
    ert_workflow_list_type *workflow_list, const char *path) {
    DIR *dirH = opendir(path);
    std::set<std::string> names;
    if (!dirH) {
        fprintf(stderr,
                "** Warning: failed to open workflow/jobs directory: %s\n",
                path);
        return;
    }
    while (true) {
        struct dirent *entry = readdir(dirH);
        if (entry == NULL)
            break;

        if ((strcmp(entry->d_name, ".") == 0) ||
            (strcmp(entry->d_name, "..") == 0))
            continue;

        char *root_name = entry->d_name;
        if (names.count(root_name))
            continue;

        char *full_path = ert_workflow_list_alloc_name(path, root_name);
        if (!full_path)
            continue;

        names.insert(root_name);
        logger->info("Adding workflow job:{}", full_path);
        ert_workflow_list_add_job(workflow_list, root_name, full_path);
        free(full_path);
    }
    closedir(dirH);
}

stringlist_type *
ert_workflow_list_get_job_names(const ert_workflow_list_type *workflow_list) {
    return workflow_joblist_get_job_names(workflow_list->joblist);
}

static void ert_workflow_list_init(ert_workflow_list_type *workflow_list,
                                   const config_content_type *config) {
    /* Adding jobs */
    {
        if (config_content_has_item(config, WORKFLOW_JOB_DIRECTORY_KEY)) {
            const config_content_item_type *jobpath_item =
                config_content_get_item(config, WORKFLOW_JOB_DIRECTORY_KEY);
            for (int i = 0; i < config_content_item_get_size(jobpath_item);
                 i++) {
                config_content_node_type *path_node =
                    config_content_item_iget_node(jobpath_item, i);

                for (int j = 0; j < config_content_node_get_size(path_node);
                     j++)
                    ert_workflow_list_add_jobs_in_directory(
                        workflow_list,
                        config_content_node_iget_as_abspath(path_node, j));
            }
        }
    }

    {
        if (config_content_has_item(config, LOAD_WORKFLOW_JOB_KEY)) {
            const config_content_item_type *job_item =
                config_content_get_item(config, LOAD_WORKFLOW_JOB_KEY);
            for (int i = 0; i < config_content_item_get_size(job_item); i++) {
                config_content_node_type *job_node =
                    config_content_item_iget_node(job_item, i);
                const char *config_file =
                    config_content_node_iget_as_path(job_node, 0);
                const char *job_name =
                    config_content_node_safe_iget(job_node, 1);
                ert_workflow_list_add_job(workflow_list, job_name, config_file);
            }
        }
    }

    /* Adding workflows */
    {
        if (config_content_has_item(config, LOAD_WORKFLOW_KEY)) {
            const config_content_item_type *workflow_item =
                config_content_get_item(config, LOAD_WORKFLOW_KEY);
            for (int i = 0; i < config_content_item_get_size(workflow_item);
                 i++) {
                config_content_node_type *workflow_node =
                    config_content_item_iget_node(workflow_item, i);

                const char *workflow_file =
                    config_content_node_iget_as_abspath(workflow_node, 0);
                const char *workflow_name =
                    config_content_node_safe_iget(workflow_node, 1);

                ert_workflow_list_add_workflow(workflow_list, workflow_file,
                                               workflow_name);
            }
        }
    }

    if (config_content_has_item(config, HOOK_WORKFLOW_KEY)) {
        for (int ihook = 0;
             ihook < config_content_get_occurences(config, HOOK_WORKFLOW_KEY);
             ihook++) {
            const char *workflow_name =
                config_content_iget(config, HOOK_WORKFLOW_KEY, ihook, 0);
            hook_run_mode_enum run_mode = hook_workflow_run_mode_from_name(
                config_content_iget(config, HOOK_WORKFLOW_KEY, ihook, 1));
            if (ert_workflow_list_has_workflow(workflow_list, workflow_name)) {
                workflow_type *workflow = ert_workflow_list_get_workflow(
                    workflow_list, workflow_name);
                hook_workflow_type *hook =
                    hook_workflow_alloc(workflow, run_mode);
                vector_append_owned_ref(workflow_list->hook_workflow_list, hook,
                                        hook_workflow_free__);
            } else {
                fprintf(stderr,
                        "** Warning: While hooking workflow: %s not recognized "
                        "among "
                        "the list of loaded workflows.",
                        workflow_name);
            }
        }
    }
}

const hook_workflow_type *ert_workflow_list_iget_hook_workflow(
    const ert_workflow_list_type *workflow_list, int index) {
    return (hook_workflow_type *)vector_iget(workflow_list->hook_workflow_list,
                                             index);
}

int ert_workflow_list_num_hook_workflows(
    const ert_workflow_list_type *workflow_list) {
    return vector_get_size(workflow_list->hook_workflow_list);
}

workflow_type *
ert_workflow_list_get_workflow(ert_workflow_list_type *workflow_list,
                               const char *workflow_name) {
    const char *lookup_name = workflow_name;

    if (hash_has_key(workflow_list->alias_map, workflow_name))
        lookup_name =
            (const char *)hash_get(workflow_list->alias_map, workflow_name);

    return (workflow_type *)hash_get(workflow_list->workflows, lookup_name);
}

bool ert_workflow_list_has_workflow(ert_workflow_list_type *workflow_list,
                                    const char *workflow_name) {
    return hash_has_key(workflow_list->workflows, workflow_name) ||
           hash_has_key(workflow_list->alias_map, workflow_name);
}
stringlist_type *
ert_workflow_list_alloc_namelist(ert_workflow_list_type *workflow_list) {
    return hash_alloc_stringlist(workflow_list->workflows);
}

int ert_workflow_list_get_size(const ert_workflow_list_type *workflow_list) {
    return hash_get_size(workflow_list->workflows) +
           hash_get_size(workflow_list->alias_map);
}
