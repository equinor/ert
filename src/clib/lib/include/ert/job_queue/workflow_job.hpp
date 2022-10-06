#ifndef ERT_WORKFLOW_JOB_H
#define ERT_WORKFLOW_JOB_H

#include <ert/config/config_parser.hpp>

typedef void *(workflow_job_ftype)(void *self, const stringlist_type *arg);
typedef struct workflow_job_struct workflow_job_type;

extern "C" const char *
workflow_job_get_name(const workflow_job_type *workflow_job);
extern "C" PY_USED bool
workflow_job_internal(const workflow_job_type *workflow_job);
extern "C" config_parser_type *workflow_job_alloc_config();
extern "C" workflow_job_type *workflow_job_alloc(const char *name,
                                                 bool internal);
extern "C" void workflow_job_free(workflow_job_type *workflow_job);
void workflow_job_free__(void *arg);
void workflow_job_set_executable(workflow_job_type *workflow_job,
                                 const char *executable);
extern "C" workflow_job_type *
workflow_job_config_alloc(const char *name, config_parser_type *config,
                          const char *config_file);

void workflow_job_update_config_compiler(const workflow_job_type *workflow_job,
                                         config_parser_type *config_compiler);
void workflow_job_set_executable(workflow_job_type *workflow_job,
                                 const char *executable);
extern "C" PY_USED char *
workflow_job_get_executable(workflow_job_type *workflow_job);

void workflow_job_set_internal_script(workflow_job_type *workflow_job,
                                      const char *script_path);
extern "C" PY_USED char *
workflow_job_get_internal_script_path(const workflow_job_type *workflow_job);
extern "C" bool
workflow_job_is_internal_script(const workflow_job_type *workflow_job);

void workflow_job_set_function(workflow_job_type *workflow_job,
                               const char *function);
extern "C" PY_USED char *
workflow_job_get_function(workflow_job_type *workflow_job);
void *workflow_job_run(const workflow_job_type *job, void *self, bool verbose,
                       const stringlist_type *arg);

extern "C" PY_USED int
workflow_job_get_min_arg(const workflow_job_type *workflow_job);
extern "C" PY_USED int
workflow_job_get_max_arg(const workflow_job_type *workflow_job);
extern "C" PY_USED config_item_types
workflow_job_iget_argtype(const workflow_job_type *workflow_job, int index);

#endif
