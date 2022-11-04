#ifndef ERT_EXT_JOB_H
#define ERT_EXT_JOB_H
#include <stdio.h>

#include <ert/config/config_content.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/tooling.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>

typedef struct ext_job_struct ext_job_type;

extern "C" const char *ext_job_get_help_text(const ext_job_type *job);
void ext_job_set_help_text(ext_job_type *job, const char *help_text);

ext_job_type *ext_job_alloc_copy(const ext_job_type *);
void ext_job_free_deprecated_argv(ext_job_type *ext_job); //DEPRECATED
ext_job_type *ext_job_alloc(const char *, const char *license_root_path,
                            bool private_job);
extern "C" const char *ext_job_get_name(const ext_job_type *);
extern "C" void ext_job_free(ext_job_type *);
void ext_job_free__(void *);
extern "C" void ext_job_save(const ext_job_type *ext_job);
void ext_job_fprintf(const ext_job_type *, FILE *stream);
void ext_job_set_private_arg(ext_job_type *, const char *, const char *);
extern "C" void ext_job_set_define_args(ext_job_type *ext_job,
                                        const subst_list_type *define_args);

extern "C" ext_job_type *ext_job_fscanf_alloc(const char *, const char *,
                                              bool private_job, const char *,
                                              bool search_path);

extern "C" subst_list_type *ext_job_get_private_args(ext_job_type *ext_job);
extern "C" const stringlist_type *
ext_job_get_arglist(const ext_job_type *ext_job);
extern "C" const stringlist_type *
ext_job_get_argvalues(const ext_job_type *ext_job);
bool ext_job_is_shared(const ext_job_type *ext_job);
extern "C" bool ext_job_is_private(const ext_job_type *ext_job);

extern "C" void ext_job_set_executable(ext_job_type *ext_job,
                                       const char *executable_abs,
                                       const char *executable_input,
                                       bool search_path);
extern "C" const char *ext_job_get_executable(const ext_job_type *ext_job);

extern "C" void ext_job_set_args(ext_job_type *ext_job,
                                 const stringlist_type *argv);
extern "C" void ext_job_set_config_file(ext_job_type *ext_job,
                                        const char *config_file);
extern "C" const char *ext_job_get_config_file(const ext_job_type *ext_job);
extern "C" void ext_job_set_target_file(ext_job_type *ext_job,
                                        const char *target_file);
extern "C" const char *ext_job_get_target_file(const ext_job_type *ext_job);
void ext_job_set_start_file(ext_job_type *ext_job, const char *start_file);
extern "C" const char *ext_job_get_start_file(const ext_job_type *ext_job);
extern "C" const char *ext_job_get_name(const ext_job_type *ext_job);
extern "C" void ext_job_set_stdin_file(ext_job_type *ext_job,
                                       const char *stdin_file);
extern "C" const char *ext_job_get_stdin_file(const ext_job_type *ext_job);
extern "C" void ext_job_set_stdout_file(ext_job_type *ext_job,
                                        const char *stdout_file);
extern "C" const char *ext_job_get_stdout_file(const ext_job_type *ext_job);
extern "C" void ext_job_set_stderr_file(ext_job_type *ext_job,
                                        const char *stderr_file);
extern "C" const char *ext_job_get_stderr_file(const ext_job_type *ext_job);

extern "C" PY_USED int ext_job_get_min_arg(const ext_job_type *ext_job);
extern "C" PY_USED int ext_job_get_max_arg(const ext_job_type *ext_job);
extern "C" config_item_types ext_job_iget_argtype(const ext_job_type *ext_job,
                                                  int index);

extern "C" void ext_job_set_max_running(ext_job_type *ext_job, int max_running);
extern "C" int ext_job_get_max_running(const ext_job_type *ext_job);
extern "C" void ext_job_set_max_running_minutes(ext_job_type *ext_job,
                                                int max_running_minutes);
extern "C" int ext_job_get_max_running_minutes(const ext_job_type *ext_job);
extern "C" void ext_job_add_environment(ext_job_type *ext_job, const char *key,
                                        const char *value);
extern "C" void ext_job_clear_environment(ext_job_type *ext_job);
extern "C" hash_type *ext_job_get_exec_env(ext_job_type *ext_job);
extern "C" hash_type *ext_job_get_default_mapping(ext_job_type *ext_job);
extern "C" hash_type *ext_job_get_environment(ext_job_type *ext_job);
extern "C" void ext_job_set_private_args_from_string(ext_job_type *ext_job,
                                                     const char *arg_string);
extern "C" const char *ext_job_get_license_path(const ext_job_type *);

extern "C" const char *ext_job_get_error_file(const ext_job_type *ext_job);

#endif
