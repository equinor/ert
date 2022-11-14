#ifndef ERT_WORKFLOW_LIST_H
#define ERT_WORKFLOW_LIST_H

#include <ert/res_util/subst_list.hpp>

#include <ert/config/config_content.hpp>
#include <ert/config/config_error.hpp>
#include <ert/config/config_parser.hpp>
#include <ert/enkf/hook_workflow.hpp>

#include <ert/job_queue/workflow.hpp>
#include <ert/job_queue/workflow_job.hpp>

typedef struct ert_workflow_list_struct ert_workflow_list_type;

extern "C" workflow_type *
ert_workflow_list_get_workflow(ert_workflow_list_type *workflow_list,
                               const char *workflow_name);
extern "C" workflow_type *
ert_workflow_list_add_workflow(ert_workflow_list_type *workflow_list,
                               const char *workflow_file,
                               const char *workflow_name);
extern "C" void ert_workflow_list_free(ert_workflow_list_type *workflow_list);
ert_workflow_list_type *ert_workflow_list_alloc_empty();
ert_workflow_list_type *ert_workflow_list_alloc_load_site_config(
    const subst_list_type *, const config_content_type *site_config_content);
extern "C" ert_workflow_list_type *
ert_workflow_list_alloc(const config_content_type *config_content,
                        const config_content_type *site_config_content);
extern "C" PY_USED ert_workflow_list_type *ert_workflow_list_alloc_full(
    workflow_joblist_type *workflow_joblist, const char **hook_workflow_names,
    const char **hook_workflow_run_modes, int hook_workflow_count);

void ert_workflow_list_add_jobs_in_directory(
    ert_workflow_list_type *workflow_list, const char *path);
extern "C" void ert_workflow_list_add_job(ert_workflow_list_type *workflow_list,
                                          const char *job_name,
                                          const char *config_file);
extern "C" bool
ert_workflow_list_has_job(const ert_workflow_list_type *workflow_list,
                          const char *job_name);
extern "C" const workflow_job_type *
ert_workflow_list_get_job(const ert_workflow_list_type *workflow_list,
                          const char *job_name);
extern "C" stringlist_type *
ert_workflow_list_get_job_names(const ert_workflow_list_type *workflow_list);
void ert_workflow_list_add_alias(ert_workflow_list_type *workflow_list,
                                 const char *real_name, const char *alias);
extern "C" bool
ert_workflow_list_has_workflow(ert_workflow_list_type *workflow_list,
                               const char *workflow_name);
extern "C" stringlist_type *
ert_workflow_list_alloc_namelist(ert_workflow_list_type *workflow_list);
void ert_workflow_list_set_verbose(ert_workflow_list_type *workflow_list,
                                   bool verbose);
int ert_workflow_list_get_size(const ert_workflow_list_type *workflow_list);

extern "C" const hook_workflow_type *ert_workflow_list_iget_hook_workflow(
    const ert_workflow_list_type *workflow_list, int index);

extern "C" int ert_workflow_list_num_hook_workflows(
    const ert_workflow_list_type *workflow_list);

#endif
