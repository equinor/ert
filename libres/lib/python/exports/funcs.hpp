#pragma once
#include "types.hpp"

extern "C" {

// AnalysisModule
void *analysis_module_alloc(const char *);
void analysis_module_free(analysis_module_type *);
bool analysis_module_set_var(analysis_module_type *, const char *,
                             const char *);
const char *analysis_module_get_table_name(analysis_module_type *);
const char *analysis_module_get_name(analysis_module_type *);
bool analysis_module_check_option(analysis_module_type *,
                                  analysis_module_options_enum);
bool analysis_module_has_var(analysis_module_type *, const char *);
double analysis_module_get_double(analysis_module_type *, const char *);
int analysis_module_get_int(analysis_module_type *, const char *);
bool analysis_module_get_bool(analysis_module_type *, const char *);
const char *analysis_module_get_ptr(analysis_module_type *, const char *);
void analysis_module_init_update(analysis_module_type *, bool_vector_type *,
                                 bool_vector_type *, matrix_type *,
                                 matrix_type *, matrix_type *, matrix_type *,
                                 matrix_type *, rng_type *);
void analysis_module_updateA(analysis_module_type *, matrix_type *,
                             matrix_type *, matrix_type *, matrix_type *,
                             matrix_type *, matrix_type *, void *, rng_type *);
void analysis_module_initX(analysis_module_type *, matrix_type *, matrix_type *,
                           matrix_type *, matrix_type *, matrix_type *,
                           matrix_type *, matrix_type *, rng_type *);
// ConfigError
void config_error_free(config_error_type *);
int config_error_count(config_error_type *);
const char *config_error_iget(config_error_type *, int);
// ConfigPathElm
void config_path_elm_free(config_path_elm_type *);
const char *config_path_elm_get_relpath(config_path_elm_type *);
const char *config_path_elm_get_abspath(config_path_elm_type *);
// ContentNode
const char *config_content_node_iget(content_node_type *, int);
int config_content_node_get_size(content_node_type *);
const char *config_content_node_get_full_string(content_node_type *,
                                                const char *);
config_content_type_enum config_content_node_iget_type(content_node_type *,
                                                       int);
const char *config_content_node_iget_as_abspath(content_node_type *, int);
const char *config_content_node_iget_as_relpath(content_node_type *, int);
int config_content_node_iget_as_int(content_node_type *, int);
double config_content_node_iget_as_double(content_node_type *, int);
const char *config_content_node_iget_as_path(content_node_type *, int);
bool config_content_node_iget_as_bool(content_node_type *, int);
time_t config_content_node_iget_as_isodate(content_node_type *, int);
// ContentItem
void *config_content_item_alloc(schema_item_type *, void *);
int config_content_item_get_size(content_item_type *);
content_node_type *config_content_item_iget_node(content_item_type *, int);
void config_content_item_free(content_item_type *);
// ContentTypeEnum
bool config_schema_item_valid_string(config_content_type_enum, const char*, bool);
// ConfigContent
void *config_content_alloc(const char *);
void config_content_free(config_content_type *);
bool config_content_is_valid(config_content_type *);
bool config_content_has_item(config_content_type *, const char *);
content_item_type *config_content_get_item(config_content_type *, const char *);
config_error_type *config_content_get_errors(config_content_type *);
stringlist_type *config_content_get_warnings(config_content_type *);
const char *config_content_get_config_path(config_content_type *);
config_path_elm_type *config_content_add_path_elm(config_content_type *,
                                                  const char *);
void config_content_add_define(config_content_type *, const char *,
                               const char *);
int config_content_get_size(config_content_type *);
stringlist_type *config_content_alloc_keys(config_content_type *);
// ConfigParser
void *config_alloc();
schema_item_type *config_add_schema_item(config_parser_type *, const char *,
                                         bool);
void config_free(config_parser_type *);
config_content_type *config_parse(config_parser_type *, const char *,
                                  const char *, const char *, const char *,
                                  hash_type *, config_unrecognized_enum, bool);
int config_get_schema_size(config_parser_type *);
schema_item_type *config_get_schema_item(config_parser_type *, const char *);
bool config_has_schema_item(config_parser_type *, const char *);
bool config_parser_add_key_values(config_parser_type *, config_content_type *,
                                  const char *, stringlist_type *,
                                  config_path_elm_type *, const char *,
                                  config_unrecognized_enum);
void config_validate(config_parser_type *, config_content_type *);
// ConfigSettings
void *config_settings_alloc(const char *);
void config_settings_free(config_settings_type *);
bool config_settings_add_setting(config_settings_type *, const char *,
                                 config_content_type_enum, const char *);
void config_settings_add_double_setting(config_settings_type *, const char *,
                                        double);
void config_settings_add_int_setting(config_settings_type *, const char *, int);
void config_settings_add_string_setting(config_settings_type *, const char *,
                                        const char *);
void config_settings_add_bool_setting(config_settings_type *, const char *,
                                      bool);
bool config_settings_has_key(config_settings_type *, const char *);
config_content_type_enum config_settings_get_value_type(config_settings_type *,
                                                        const char *);
void config_settings_init_parser(config_settings_type *, config_parser_type *,
                                 bool);
void config_settings_apply(config_settings_type *, config_content_type *);
stringlist_type *config_settings_alloc_keys(config_settings_type *);
const char *config_settings_get_value(config_settings_type *, const char *);
int config_settings_get_int_value(config_settings_type *, const char *);
double config_settings_get_double_value(config_settings_type *, const char *);
bool config_settings_get_bool_value(config_settings_type *, const char *);
bool config_settings_set_value(config_settings_type *, const char *,
                               const char *);
bool config_settings_set_int_value(config_settings_type *, const char *, int);
bool config_settings_set_double_value(config_settings_type *, const char *,
                                      double);
bool config_settings_set_bool_value(config_settings_type *, const char *, bool);
// SchemaItem
void *config_schema_item_alloc(const char *, bool);
void config_schema_item_free(schema_item_type *);
config_content_type_enum config_schema_item_iget_type(schema_item_type *, int);
void config_schema_item_iset_type(schema_item_type *, int,
                                  config_content_type_enum);
void config_schema_item_set_argc_minmax(schema_item_type *, int, int);
void config_schema_item_add_indexed_alternative(schema_item_type *, int,
                                                const char *);
void config_schema_item_set_deprecated(schema_item_type *, const char *);
// ErtTemplate
void ert_template_free(ert_template_type *);
const char *ert_template_get_template_file(ert_template_type *);
const char *ert_template_get_target_file(ert_template_type *);
subst_list_type *ert_template_get_arg_list(ert_template_type *);
// AnalysisConfig
void *analysis_config_alloc(config_content_type *);
void *analysis_config_alloc_load(const char *);
void *analysis_config_alloc_full(double, bool, int, const char *, double, bool,
                                 bool, double, int, int);
void analysis_config_add_module_copy(analysis_config_type *, const char *,
                                     const char *);
void analysis_config_free(analysis_config_type *);
int analysis_config_get_rerun(analysis_config_type *);
void analysis_config_set_rerun(analysis_config_type *, bool);
int analysis_config_get_rerun_start(analysis_config_type *);
void analysis_config_set_rerun_start(analysis_config_type *, int);
const char *analysis_config_get_log_path(analysis_config_type *);
void analysis_config_set_log_path(analysis_config_type *, const char *);
analysis_iter_config_type *
analysis_config_get_iter_config(analysis_config_type *);
int analysis_config_get_max_runtime(analysis_config_type *);
void analysis_config_set_max_runtime(analysis_config_type *, int);
bool analysis_config_get_stop_long_running(analysis_config_type *);
void analysis_config_set_stop_long_running(analysis_config_type *, bool);
const char *analysis_config_get_active_module_name(analysis_config_type *);
analysis_module_type *analysis_config_get_module(analysis_config_type *,
                                                 const char *);
bool analysis_config_select_module(analysis_config_type *, const char *);
bool analysis_config_has_module(analysis_config_type *, const char *);
double analysis_config_get_alpha(analysis_config_type *);
void analysis_config_set_alpha(analysis_config_type *, double);
double analysis_config_get_std_cutoff(analysis_config_type *);
void analysis_config_set_std_cutoff(analysis_config_type *, double);
void analysis_config_set_global_std_scaling(analysis_config_type *, double);
double analysis_config_get_global_std_scaling(analysis_config_type *);
int analysis_config_get_min_realisations(analysis_config_type *);
// AnalysisIterConfig
void *analysis_iter_config_alloc();
void *analysis_iter_config_alloc_full(const char *, int, int);
void analysis_iter_config_free(analysis_iter_config_type *);
void analysis_iter_config_set_num_iterations(analysis_iter_config_type *, int);
int analysis_iter_config_get_num_iterations(analysis_iter_config_type *);
int analysis_iter_config_get_num_retries_per_iteration(
    analysis_iter_config_type *);
bool analysis_iter_config_num_iterations_set(analysis_iter_config_type *);
void analysis_iter_config_set_case_fmt(analysis_iter_config_type *,
                                       const char *);
const char *analysis_iter_config_get_case_fmt(analysis_iter_config_type *);
bool analysis_iter_config_case_fmt_set(analysis_iter_config_type *);
// EclConfig
void *ecl_config_alloc(config_content_type *);
void *ecl_config_alloc_full(bool, const char *, ecl_grid_type *, const char *,
                            stringlist_type *, time_t, const char *);
void ecl_config_free(ecl_config_type *);
const char *ecl_config_get_data_file(ecl_config_type *);
void ecl_config_set_data_file(ecl_config_type *, const char *);
ui_return_type *ecl_config_validate_data_file(ecl_config_type *, const char *);
const char *ecl_config_get_gridfile(ecl_config_type *);
void ecl_config_set_grid(ecl_config_type *, const char *);
ui_return_type *ecl_config_validate_grid(ecl_config_type *, const char *);
ecl_grid_type *ecl_config_get_grid(ecl_config_type *);
const char *ecl_config_get_refcase_name(ecl_config_type *);
ecl_sum_type *ecl_config_get_refcase(ecl_config_type *);
void ecl_config_load_refcase(ecl_config_type *, const char *);
ui_return_type *ecl_config_validate_refcase(ecl_config_type *, const char *);
bool ecl_config_has_refcase(ecl_config_type *);
const char *ecl_config_get_depth_unit(ecl_config_type *);
const char *ecl_config_get_pressure_unit(ecl_config_type *);
bool ecl_config_active(ecl_config_type *);
int ecl_config_get_last_history_restart(ecl_config_type *);
time_t ecl_config_get_end_date(ecl_config_type *);
int ecl_config_get_num_cpu(ecl_config_type *);
// EnkfFs
void *enkf_fs_mount(const char *);
void enkf_fs_sync(enkf_fs_type *);
bool enkf_fs_exists(const char *);
int enkf_fs_disk_version(const char *);
bool enkf_fs_update_disk_version(const char *, int, int);
int enkf_fs_decref(enkf_fs_type *);
int enkf_fs_incref(enkf_fs_type *);
int enkf_fs_get_refcount(enkf_fs_type *);
const char *enkf_fs_get_case_name(enkf_fs_type *);
bool enkf_fs_is_read_only(enkf_fs_type *);
bool enkf_fs_is_running(enkf_fs_type *);
void enkf_fs_fsync(enkf_fs_type *);
enkf_fs_type *enkf_fs_create_fs(const char *, enkf_fs_type_enum, void *, bool);
time_map_type *enkf_fs_get_time_map(enkf_fs_type *);
state_map_type *enkf_fs_get_state_map(enkf_fs_type *);
summary_key_set_type *enkf_fs_get_summary_key_set(enkf_fs_type *);
// EnkfFsManager
enkf_fs_type *enkf_main_get_fs_ref(enkf_fs_manager_type *);
void enkf_main_set_fs(enkf_fs_manager_type *, enkf_fs_type *, const char *);
stringlist_type *enkf_main_alloc_caselist(enkf_fs_manager_type *);
bool enkf_main_case_is_initialized(enkf_fs_manager_type *, const char *,
                                   bool_vector_type *);
void enkf_main_init_case_from_existing(enkf_fs_manager_type *, enkf_fs_type *,
                                       int, enkf_fs_type *);
void enkf_main_init_current_case_from_existing(enkf_fs_manager_type *,
                                               enkf_fs_type *, int);
state_map_type *enkf_main_alloc_readonly_state_map(enkf_fs_manager_type *,
                                                   const char *);
// EnKFMain
void *enkf_main_alloc(res_config_type *, bool, bool);
void enkf_main_free(enkf_main_type *);
queue_config_type *enkf_main_get_queue_config(enkf_main_type *);
int enkf_main_get_ensemble_size(enkf_main_type *);
ens_config_type *enkf_main_get_ensemble_config(enkf_main_type *);
model_config_type *enkf_main_get_model_config(enkf_main_type *);
local_config_type *enkf_main_get_local_config(enkf_main_type *);
analysis_config_type *enkf_main_get_analysis_config(enkf_main_type *);
site_config_type *enkf_main_get_site_config(enkf_main_type *);
ecl_config_type *enkf_main_get_ecl_config(enkf_main_type *);
const char *enkf_main_get_schedule_prediction_file(enkf_main_type *);
subst_list_type *enkf_main_get_data_kw(enkf_main_type *);
void enkf_main_clear_data_kw(enkf_main_type *);
void enkf_main_add_data_kw(enkf_main_type *, const char *, const char *);
void enkf_main_resize_ensemble(enkf_main_type *, int);
enkf_obs_type *enkf_main_get_obs(enkf_main_type *);
bool enkf_main_load_obs(enkf_main_type *, const char *, bool);
ert_templates_type *enkf_main_get_templates(enkf_main_type *);
const char *enkf_main_get_site_config_file(enkf_main_type *);
int enkf_main_get_history_length(enkf_main_type *);
int enkf_main_get_observation_count(enkf_main_type *, const char *);
bool enkf_main_have_obs(enkf_main_type *);
enkf_state_type *enkf_main_iget_state(enkf_main_type *, int);
ert_workflow_list_type *enkf_main_get_workflow_list(enkf_main_type *);
hook_manager_type *enkf_main_get_hook_manager(enkf_main_type *);
const char *enkf_main_get_user_config_file(enkf_main_type *);
const char *enkf_main_get_mount_root(enkf_main_type *);
bool enkf_main_export_field_with_fs(enkf_main_type *, const char *,
                                    const char *, bool_vector_type *,
                                    enkf_field_file_format_enum, int,
                                    enkf_fs_manager_type *);
int enkf_main_load_from_forward_model_from_gui(enkf_main_type *, int,
                                               bool_vector_type *,
                                               enkf_fs_type *);
int enkf_main_load_from_run_context_from_gui(enkf_main_type *,
                                             ert_run_context_type *,
                                             enkf_fs_type *);
void enkf_main_create_run_path(enkf_main_type *, ert_run_context_type *);
ert_run_context_type *enkf_main_alloc_ert_run_context_ENSEMBLE_EXPERIMENT(
    enkf_main_type *, enkf_fs_type *, bool_vector_type *, int);
runpath_list_type *enkf_main_get_runpath_list(enkf_main_type *);
runpath_list_type *enkf_main_alloc_runpath_list(enkf_main_type *);
void enkf_main_add_node(enkf_main_type *, enkf_config_node_type *);
res_config_type *enkf_main_get_res_config(enkf_main_type *);
void enkf_main_init_run(enkf_main_type *, ert_run_context_type *);
rng_type *enkf_main_get_shared_rng(enkf_main_type *);
// EnkfObs
void *enkf_obs_alloc(history_type *, time_map_type *, ecl_grid_type *,
                     ecl_sum_type *, ens_config_type *);
void enkf_obs_free(enkf_obs_type *);
int enkf_obs_get_size(enkf_obs_type *);
bool enkf_obs_is_valid(enkf_obs_type *);
void enkf_obs_load(enkf_obs_type *, const char *, double);
void enkf_obs_clear(enkf_obs_type *);
stringlist_type *enkf_obs_alloc_typed_keylist(enkf_obs_type *,
                                              enkf_obs_impl_type);
stringlist_type *enkf_obs_alloc_matching_keylist(enkf_obs_type *, const char *);
bool enkf_obs_has_key(enkf_obs_type *, const char *);
enkf_obs_impl_type enkf_obs_get_type(enkf_obs_type *, const char *);
obs_vector_type *enkf_obs_get_vector(enkf_obs_type *, const char *);
obs_vector_type *enkf_obs_iget_vector(enkf_obs_type *, int);
time_t enkf_obs_iget_obs_time(enkf_obs_type *, int);
void enkf_obs_add_obs_vector(enkf_obs_type *, obs_vector_type *);
void enkf_obs_get_obs_and_measure_data(enkf_obs_type *, enkf_fs_type *,
                                       local_obsdata_type *, int_vector_type *,
                                       meas_data_type *, obs_data_type *);
local_obsdata_type *enkf_obs_alloc_all_active_local_obs(enkf_obs_type *,
                                                        const char *);
void enkf_obs_local_scale_std(enkf_obs_type *, local_obsdata_type *, double);
// EnKFState
void enkf_state_free(enkf_state_type *);
ens_config_type *enkf_state_get_ensemble_config(enkf_state_type *);
void enkf_state_initialize(enkf_state_type *, enkf_fs_type *, stringlist_type *,
                           enkf_init_mode_enum);
bool enkf_state_complete_forward_modelOK(res_config_type *, run_arg_type *);
bool enkf_state_complete_forward_model_EXIT_handler__(run_arg_type *);
// EnsembleConfig
void *ensemble_config_alloc(config_content_type *, ecl_grid_type *,
                            ecl_sum_type *);
void *ensemble_config_alloc_full(const char *);
void ensemble_config_free(ens_config_type *);
bool ensemble_config_has_key(ens_config_type *, const char *);
int ensemble_config_get_size(ens_config_type *);
enkf_config_node_type *ensemble_config_get_node(ens_config_type *,
                                                const char *);
stringlist_type *ensemble_config_alloc_keylist(ens_config_type *);
enkf_config_node_type *ensemble_config_add_summary(ens_config_type *,
                                                   const char *, int);
enkf_config_node_type *ensemble_config_add_gen_kw(ens_config_type *,
                                                  const char *);
enkf_config_node_type *ensemble_config_add_field(ens_config_type *,
                                                 const char *, ecl_grid_type *);
stringlist_type *
ensemble_config_alloc_keylist_from_impl_type(ens_config_type *,
                                             ert_impl_type_enum);
void ensemble_config_add_node(ens_config_type *, enkf_config_node_type *);
summary_key_matcher_type *
ensemble_config_get_summary_key_matcher(ens_config_type *);
void *ensemble_config_get_trans_table(ens_config_type *);
void ensemble_config_init_SUMMARY_full(ens_config_type *, const char *,
                                       ecl_sum_type *);
// ErtRunContext
void *ert_run_context_alloc(enkf_run_mode_enum, enkf_init_mode_enum,
                            enkf_fs_type *, enkf_fs_type *, bool_vector_type *,
                            path_fmt_type *, const char *, subst_list_type *,
                            int);
ert_run_context_type *
ert_run_context_alloc_ENSEMBLE_EXPERIMENT(enkf_fs_type *, bool_vector_type *,
                                          path_fmt_type *, const char *,
                                          subst_list_type *, int);
ert_run_context_type *
ert_run_context_alloc_SMOOTHER_RUN(enkf_fs_type *, enkf_fs_type *,
                                   bool_vector_type *, path_fmt_type *,
                                   const char *, subst_list_type *, int);
ert_run_context_type *ert_run_context_alloc_SMOOTHER_UPDATE(enkf_fs_type *,
                                                            enkf_fs_type *);
ert_run_context_type *ert_run_context_alloc_CASE_INIT(enkf_fs_type *,
                                                      bool_vector_type *);
stringlist_type *ert_run_context_alloc_runpath_list(bool_vector_type *,
                                                    path_fmt_type *,
                                                    subst_list_type *, int);
const char *ert_run_context_alloc_runpath(int, path_fmt_type *,
                                          subst_list_type *, int);
int ert_run_context_get_size(ert_run_context_type *);
void ert_run_context_free(ert_run_context_type *);
bool ert_run_context_iactive(ert_run_context_type *, int);
run_arg_type *ert_run_context_iget_arg(ert_run_context_type *, int);
const char *ert_run_context_get_id(ert_run_context_type *);
bool_vector_type *ert_run_context_alloc_iactive(ert_run_context_type *);
int ert_run_context_get_iter(ert_run_context_type *);
enkf_fs_type *ert_run_context_get_update_target_fs(ert_run_context_type *);
enkf_fs_type *ert_run_context_get_sim_fs(ert_run_context_type *);
enkf_init_mode_enum ert_run_context_get_init_mode(ert_run_context_type *);
int ert_run_context_get_step1(ert_run_context_type *);
void ert_run_context_deactivate_realization(ert_run_context_type *, int);
// ErtTemplates
void *ert_templates_alloc(subst_list_type *, config_content_type *);
void *ert_templates_alloc_default(subst_list_type *);
void ert_templates_free(ert_templates_type *);
stringlist_type *ert_templates_alloc_list(ert_templates_type *);
ert_template_type *ert_templates_get_template(ert_templates_type *,
                                              const char *);
void ert_templates_clear(ert_templates_type *);
ert_template_type *ert_templates_add_template(ert_templates_type *,
                                              const char *, const char *,
                                              const char *, const char *);
// ErtWorkflowList
void *ert_workflow_list_alloc(subst_list_type *, config_content_type *);
void *ert_workflow_list_alloc_full(subst_list_type *, workflow_joblist_type *);
void ert_workflow_list_free(ert_workflow_list_type *);
stringlist_type *ert_workflow_list_alloc_namelist(ert_workflow_list_type *);
bool ert_workflow_list_has_workflow(ert_workflow_list_type *, const char *);
workflow_type *ert_workflow_list_get_workflow(ert_workflow_list_type *,
                                              const char *);
workflow_type *ert_workflow_list_add_workflow(ert_workflow_list_type *,
                                              const char *, const char *);
subst_list_type *ert_workflow_list_get_context(ert_workflow_list_type *);
void ert_workflow_list_add_job(ert_workflow_list_type *, const char *,
                               const char *);
bool ert_workflow_list_has_job(ert_workflow_list_type *, const char *);
workflow_job_type *ert_workflow_list_get_job(ert_workflow_list_type *,
                                             const char *);
stringlist_type *ert_workflow_list_get_job_names(ert_workflow_list_type *);
// ESUpdate
bool enkf_main_smoother_update(es_update_type *, enkf_fs_type *,
                               enkf_fs_type *);
// ForwardLoadContext
void *forward_load_context_alloc(run_arg_type *, bool, ecl_config_type *,
                                 const char *, stringlist_type *);
void forward_load_context_select_step(forward_load_context_type *, int);
int forward_load_context_get_load_step(forward_load_context_type *);
void forward_load_context_free(forward_load_context_type *);
// HookManager
void *hook_manager_alloc(ert_workflow_list_type *, config_content_type *);
void *hook_manager_alloc_full(ert_workflow_list_type *, const char *);
void hook_manager_free(hook_manager_type *);
const char *hook_manager_get_runpath_list_file(hook_manager_type *);
runpath_list_type *hook_manager_get_runpath_list(hook_manager_type *);
hook_workflow_type *hook_manager_iget_hook_workflow(hook_manager_type *, int);
int hook_manager_get_size(hook_manager_type *);
// HookWorkflow
workflow_type *hook_workflow_get_workflow(hook_workflow_type *);
hook_runtime_enum hook_workflow_get_run_mode(hook_workflow_type *);
// LocalConfig
void local_config_free(local_config_type *);
void local_config_clear(local_config_type *);
void local_config_clear_active(local_config_type *);
local_ministep_type *local_config_alloc_ministep(local_config_type *,
                                                 const char *,
                                                 analysis_module_type *);
void local_updatestep_add_ministep(local_updatestep_type *,
                                   local_ministep_type *);
void local_config_alloc_obsdata(local_config_type *, const char *);
bool local_config_has_obsdata(local_config_type *, const char *);
local_updatestep_type *local_config_get_updatestep(local_config_type *);
local_ministep_type *local_config_get_ministep(local_config_type *,
                                               const char *);
local_obsdata_type *local_config_get_obsdata(local_config_type *, const char *);
local_obsdata_type *local_config_alloc_obsdata_copy(local_config_type *,
                                                    const char *, const char *);
// LocalMinistep
void local_ministep_add_obsdata_node(local_ministep_type *,
                                     local_obsdata_node_type *);
local_obsdata_type *local_ministep_get_obsdata(local_ministep_type *);
obs_data_type *local_ministep_get_obs_data(local_ministep_type *);
void local_ministep_free(local_ministep_type *);
void local_ministep_add_obsdata(local_ministep_type *, local_obsdata_type *);
const char *local_ministep_get_name(local_ministep_type *);
bool local_ministep_data_is_active(local_ministep_type*, const char*);
void local_ministep_activate_data(local_ministep_type*, const char*);
row_scaling_type* local_ministep_get_or_create_row_scaling(local_ministep_type*, const char*);
// LocalObsdata
void *local_obsdata_alloc(const char *);
void local_obsdata_free(local_obsdata_type *);
int local_obsdata_get_size(local_obsdata_type *);
bool local_obsdata_has_node(local_obsdata_type *, const char *);
bool local_obsdata_add_node(local_obsdata_type *, local_obsdata_node_type *);
void local_obsdata_del_node(local_obsdata_type *, const char *);
const char *local_obsdata_get_name(local_obsdata_type *);
local_obsdata_node_type *local_obsdata_iget(local_obsdata_type *, int);
local_obsdata_node_type *local_obsdata_get(local_obsdata_type *, const char *);
active_list_type *local_obsdata_get_copy_node_active_list(local_obsdata_type *,
                                                          const char *);
active_list_type *local_obsdata_get_node_active_list(local_obsdata_type *,
                                                     const char *);
// LocalObsdataNode
void *local_obsdata_node_alloc(const char *, bool);
void local_obsdata_node_free(local_obsdata_node_type *);
const char *local_obsdata_node_get_key(local_obsdata_node_type *);
void local_obsdata_node_add_tstep(local_obsdata_node_type *, int);
bool local_obsdata_node_tstep_active(local_obsdata_node_type *, int);
bool local_obsdata_node_all_timestep_active(local_obsdata_node_type *);
void local_obsdata_node_set_all_timestep_active(local_obsdata_node_type *,
                                                bool);
active_list_type *local_obsdata_node_get_active_list(local_obsdata_node_type *);
// LocalUpdateStep
int local_updatestep_get_num_ministep(local_updatestep_type *);
local_ministep_type *local_updatestep_iget_ministep(local_updatestep_type *,
                                                    int);
void local_updatestep_free(local_updatestep_type *);
const char *local_updatestep_get_name(local_updatestep_type *);
// LogConfig
void *log_config_alloc(config_content_type *);
void *log_config_alloc_load(const char *);
void *log_config_alloc_full(const char *, message_level_enum);
void log_config_free(log_config_type *);
const char *log_config_get_log_file(log_config_type *);
message_level_enum log_config_get_log_level(log_config_type *);
// MeasBlock
void *meas_block_alloc(const char *, bool_vector_type *, int);
void meas_block_free(meas_block_type *);
int meas_block_get_active_ens_size(meas_block_type *);
int meas_block_get_total_ens_size(meas_block_type *);
int meas_block_get_total_obs_size(meas_block_type *);
double meas_block_iget(meas_block_type *, int, int);
void meas_block_iset(meas_block_type *, int, int, double);
double meas_block_iget_ens_mean(meas_block_type *, int);
double meas_block_iget_ens_std(meas_block_type *, int);
bool meas_block_iens_active(meas_block_type *, int);
// MeasData
void *meas_data_alloc(bool_vector_type *);
void meas_data_free(meas_data_type *);
int meas_data_get_active_obs_size(meas_data_type *);
int meas_data_get_active_ens_size(meas_data_type *);
int meas_data_get_total_ens_size(meas_data_type *);
int meas_data_get_num_blocks(meas_data_type *);
bool meas_data_has_block(meas_data_type *, const char *);
meas_block_type *meas_data_get_block(meas_data_type *, const char *);
matrix_type *meas_data_allocS(meas_data_type *);
meas_block_type *meas_data_add_block(meas_data_type *, const char *, int, int);
meas_block_type *meas_data_iget_block(meas_data_type *, int);
void enkf_analysis_deactivate_std_zero(obs_data_type *, meas_data_type *);
// ModelConfig
void *model_config_alloc(config_content_type *, const char *,
                         ext_joblist_type *, int, ecl_sum_type *);
void *model_config_alloc_full(int, int, const char *, const char *,
                              const char *, const char *, forward_model_type *,
                              const char *, time_map_type *, const char *,
                              const char *, history_source_enum,
                              ext_joblist_type *, ecl_sum_type *);
void model_config_free(model_config_type *);
forward_model_type *model_config_get_forward_model(model_config_type *);
int model_config_get_max_internal_submit(model_config_type *);
const char *model_config_get_runpath_as_char(model_config_type *);
bool model_config_select_runpath(model_config_type *, const char *);
void model_config_set_runpath(model_config_type *, const char *);
const char *model_config_get_enspath(model_config_type *);
history_source_enum model_config_get_history_source(model_config_type *);
bool model_config_select_history(model_config_type *, history_source_enum,
                                 ecl_sum_type *);
bool model_config_has_history(model_config_type *);
const char *model_config_get_gen_kw_export_name(model_config_type *);
bool model_config_runpath_requires_iter(model_config_type *);
const char *model_config_get_jobname_fmt(model_config_type *);
path_fmt_type *model_config_get_runpath_fmt(model_config_type *);
int model_config_get_num_realizations(model_config_type *);
const char *model_config_get_obs_config_file(model_config_type *);
const char *model_config_get_data_root(model_config_type *);
void model_config_get_external_time_map(model_config_type *);
// ObsBlock
void *obs_block_alloc(const char *, int, matrix_type *, bool, double);
void obs_block_free(obs_block_type *);
int obs_block_get_size(obs_block_type *);
int obs_block_get_active_size(obs_block_type *);
void obs_block_iset(obs_block_type *, int, double, double);
double obs_block_iget_value(obs_block_type *, int);
double obs_block_iget_std(obs_block_type *, int);
const char *obs_block_get_key(obs_block_type *);
bool obs_block_iget_is_active(obs_block_type *, int);
// ObsData
void *obs_data_alloc(double);
void obs_data_free(obs_data_type *);
int obs_data_get_total_size(obs_data_type *);
void obs_data_scale(obs_data_type *, matrix_type *, matrix_type *,
                    matrix_type *, matrix_type *, matrix_type *);
void obs_data_scale_matrix(obs_data_type *, matrix_type *);
void obs_data_scale_Rmatrix(obs_data_type *, matrix_type *);
double obs_data_iget_value(obs_data_type *, int);
double obs_data_iget_std(obs_data_type *, int);
obs_block_type *obs_data_add_block(obs_data_type *, const char *, int,
                                   matrix_type *, bool);
matrix_type *obs_data_allocdObs(obs_data_type *);
matrix_type *obs_data_allocR(obs_data_type *);
matrix_type *obs_data_allocD(obs_data_type *, matrix_type *, matrix_type *);
matrix_type *obs_data_allocE(obs_data_type *, rng_type *, int);
obs_block_type *obs_data_iget_block(obs_data_type *, int);
int obs_data_get_num_blocks(obs_data_type *);
// QueueConfig
void queue_config_free(queue_config_type *);
void *queue_config_alloc_load(const char *);
void *queue_config_alloc_full(const char *, bool, int, int, queue_driver_enum);
void *queue_config_alloc(config_content_type *);
queue_config_type *queue_config_alloc_local_copy(queue_config_type *);
bool queue_config_has_job_script(queue_config_type *);
const char *queue_config_get_job_script(queue_config_type *);
int queue_config_get_max_submit(queue_config_type *);
const char *queue_config_get_queue_system(queue_config_type *);
driver_type *queue_config_get_queue_driver(queue_config_type *, const char *);
int queue_config_get_num_cpu(queue_config_type *);
const char *queue_config_lsf_queue_name();
const char *queue_config_lsf_server();
const char *queue_config_lsf_resource();
const char *queue_config_lsf_driver_name();
// ResConfig
void res_config_free(res_config_type *);
void *res_config_alloc_full(const char *, const char *, subst_config_type *,
                            site_config_type *, rng_config_type *,
                            analysis_config_type *, ert_workflow_list_type *,
                            hook_manager_type *, ert_templates_type *,
                            ecl_config_type *, ens_config_type *,
                            model_config_type *, log_config_type *,
                            queue_config_type *);
config_content_type *res_config_alloc_user_content(const char *,
                                                   config_parser_type *);
const char *res_config_get_user_config_file(res_config_type *);
const char *res_config_get_config_directory(res_config_type *);
site_config_type *res_config_get_site_config(res_config_type *);
analysis_config_type *res_config_get_analysis_config(res_config_type *);
subst_config_type *res_config_get_subst_config(res_config_type *);
model_config_type *res_config_get_model_config(res_config_type *);
ecl_config_type *res_config_get_ecl_config(res_config_type *);
ens_config_type *res_config_get_ensemble_config(res_config_type *);
hook_manager_type *res_config_get_hook_manager(res_config_type *);
ert_workflow_list_type *res_config_get_workflow_list(res_config_type *);
rng_config_type *res_config_get_rng_config(res_config_type *);
ert_templates_type *res_config_get_templates(res_config_type *);
log_config_type *res_config_get_log_config(res_config_type *);
queue_config_type *res_config_get_queue_config(res_config_type *);
void res_config_init_config_parser(config_parser_type *);
// RNGConfig
void *rng_config_alloc(config_content_type *);
void *rng_config_alloc_full(const char *);
void rng_config_free(rng_config_type *);
rng_alg_type_enum rng_config_get_type(rng_config_type *);
const char *rng_config_get_random_seed(rng_config_type *);
// RowScaling
void *row_scaling_alloc();
void row_scaling_free(row_scaling_type *);
int row_scaling_get_size(row_scaling_type *);
double row_scaling_iset(row_scaling_type *, int, double);
double row_scaling_iget(row_scaling_type *, int);
double row_scaling_clamp(row_scaling_type *, double);
void row_scaling_assign_double(row_scaling_type *, const double *, int);
void row_scaling_assign_float(row_scaling_type *, const float *, int);
// RunArg
run_arg_type *run_arg_alloc_ENSEMBLE_EXPERIMENT(const char *, enkf_fs_type *,
                                                int, int, const char *,
                                                const char *,
                                                subst_list_type *);
void run_arg_free(run_arg_type *);
int run_arg_get_queue_index_safe(run_arg_type *);
void run_arg_set_queue_index(run_arg_type *, int);
bool run_arg_is_submitted(run_arg_type *);
const char *run_arg_get_run_id(run_arg_type *);
int run_arg_get_geo_id(run_arg_type *);
void run_arg_set_geo_id(run_arg_type *, int);
const char *run_arg_get_runpath(run_arg_type *);
int run_arg_get_iter(run_arg_type *);
int run_arg_get_iens(run_arg_type *);
int run_arg_get_run_status(run_arg_type *);
const char *run_arg_get_job_name(run_arg_type *);
// RunpathList
void *runpath_list_alloc(const char *);
void runpath_list_free(runpath_list_type *);
void runpath_list_add(runpath_list_type *, int, int, const char *,
                      const char *);
void runpath_list_clear(runpath_list_type *);
int runpath_list_size(runpath_list_type *);
int runpath_list_iget_iens(runpath_list_type *, int);
int runpath_list_iget_iter(runpath_list_type *, int);
const char *runpath_list_iget_runpath(runpath_list_type *, int);
const char *runpath_list_iget_basename(runpath_list_type *, int);
void runpath_list_fprintf(runpath_list_type *);
bool runpath_list_load(runpath_list_type *);
const char *runpath_list_get_export_file(runpath_list_type *);
void runpath_list_set_export_file(runpath_list_type *, const char *);
// SiteConfig
void *site_config_alloc(config_content_type *);
void *site_config_alloc_full(ext_joblist_type *, env_varlist_type *, int);
void *site_config_alloc_load_user_config(const char *);
void site_config_free(site_config_type *);
ext_joblist_type *site_config_get_installed_jobs(site_config_type *);
const char *site_config_get_license_root_path(site_config_type *);
void site_config_set_license_root_path(site_config_type *, const char *);
const char *site_config_get_location();
const char *site_config_get_config_file(site_config_type *);
int site_config_get_umask(site_config_type *);
// StateMap
void *state_map_alloc();
bool state_map_fread(state_map_type *, const char *);
void state_map_fwrite(state_map_type *, const char *);
bool state_map_equal(state_map_type *, state_map_type *);
void state_map_free(state_map_type *);
int state_map_get_size(state_map_type *);
realisation_state_enum state_map_iget(state_map_type *, int);
void state_map_iset(state_map_type *, int, realisation_state_enum);
void state_map_select_matching(state_map_type *, bool_vector_type *,
                               realisation_state_enum, bool);
bool state_map_is_readonly(state_map_type *);
bool state_map_legal_transition(realisation_state_enum, realisation_state_enum);
// SubstConfig
void *subst_config_alloc(config_content_type *);
void *subst_config_alloc_full(subst_list_type *);
void subst_config_free(subst_config_type *);
subst_list_type *subst_config_get_subst_list(subst_config_type *);
int ecl_util_get_num_cpu(const char *);
// SummaryKeyMatcher
void *summary_key_matcher_alloc();
void summary_key_matcher_free(summary_key_matcher_type *);
int summary_key_matcher_get_size(summary_key_matcher_type *);
void summary_key_matcher_add_summary_key(summary_key_matcher_type *,
                                         const char *);
bool summary_key_matcher_match_summary_key(summary_key_matcher_type *,
                                           const char *);
stringlist_type *summary_key_matcher_get_keys(summary_key_matcher_type *);
bool summary_key_matcher_summary_key_is_required(summary_key_matcher_type *,
                                                 const char *);
// SummaryKeySet
void *summary_key_set_alloc();
void *summary_key_set_alloc_from_file(const char *, bool);
void summary_key_set_free(summary_key_set_type *);
int summary_key_set_get_size(summary_key_set_type *);
bool summary_key_set_add_summary_key(summary_key_set_type *, const char *);
bool summary_key_set_has_summary_key(summary_key_set_type *, const char *);
stringlist_type *summary_key_set_alloc_keys(summary_key_set_type *);
bool summary_key_set_is_read_only(summary_key_set_type *);
void summary_key_set_fwrite(summary_key_set_type *, const char *);
// ActiveList
void *active_list_alloc();
void active_list_free(active_list_type *);
void active_list_add_index(active_list_type *, int);
int active_list_get_active_size(active_list_type *, int);
active_mode_enum active_list_get_mode(active_list_type *);
// EnkfConfigNode
enkf_config_node_type *enkf_config_node_alloc(enkf_var_type_enum,
                                              ert_impl_type_enum, bool,
                                              const char *, const char *,
                                              const char *, const char *,
                                              void *);
enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_everest(const char *, const char *,
                                        int_vector_type *);
enkf_config_node_type *enkf_config_node_alloc_summary(const char *,
                                                      load_fail_type);
enkf_config_node_type *
enkf_config_node_alloc_field(const char *, ecl_grid_type *, void *, bool);
void *enkf_config_node_get_ref(enkf_config_node_type *);
ert_impl_type_enum enkf_config_node_get_impl_type(enkf_config_node_type *);
const char *enkf_config_node_get_enkf_outfile(enkf_config_node_type *);
const char *enkf_config_node_get_min_std_file(enkf_config_node_type *);
const char *enkf_config_node_get_enkf_infile(enkf_config_node_type *);
const char *enkf_config_node_get_init_file_fmt(enkf_config_node_type *);
enkf_var_type_enum enkf_config_node_get_var_type(enkf_config_node_type *);
const char *enkf_config_node_get_key(enkf_config_node_type *);
stringlist_type *enkf_config_node_get_obs_keys(enkf_config_node_type *);
void enkf_config_node_free(enkf_config_node_type *);
bool enkf_config_node_use_forward_init(enkf_config_node_type *);
enkf_config_node_type *enkf_config_node_alloc_GEN_PARAM_full(
    const char *, bool, gen_data_file_format_type, gen_data_file_format_type,
    const char *, const char *, const char *, const char *, const char *);
enkf_config_node_type *enkf_config_node_alloc_GEN_DATA_full(
    const char *, const char *, gen_data_file_format_type, int_vector_type *,
    const char *, const char *, const char *, const char *);
enkf_config_node_type *
enkf_config_node_alloc_GEN_KW_full(const char *, bool, const char *,
                                   const char *, const char *, const char *,
                                   const char *, const char *);
enkf_config_node_type *
enkf_config_node_alloc_SURFACE_full(const char *, bool, const char *,
                                    const char *, const char *, const char *);
enkf_config_node_type *enkf_config_node_new_container(const char *);
void enkf_config_node_update_container(enkf_config_node_type *,
                                       enkf_config_node_type *);
int enkf_config_node_container_size(enkf_config_node_type *);
const char *enkf_config_node_iget_container_key(enkf_config_node_type *, int);
void enkf_config_node_update_parameter_field(
    enkf_config_node_type *, const char *, const char *, const char *,
    enkf_truncation_type_enum, double, double, const char *, const char *);
void enkf_config_node_update_general_field(enkf_config_node_type *,
                                           const char *, const char *,
                                           const char *, const char *,
                                           enkf_truncation_type_enum, double,
                                           double, const char *, const char *,
                                           const char *);
// ExtParamConfig
void *ext_param_config_alloc(const char *, stringlist_type *);
int ext_param_config_get_data_size(ext_param_config_type *);
const char *ext_param_config_iget_key(ext_param_config_type *, int);
void ext_param_config_free(ext_param_config_type *);
bool ext_param_config_has_key(ext_param_config_type *, const char *);
int ext_param_config_get_key_index(ext_param_config_type *, const char *);
int ext_param_config_ikey_get_suffix_count(ext_param_config_type *, int);
const char *ext_param_config_ikey_iget_suffix(ext_param_config_type *, int,
                                              int);
void ext_param_config_ikey_set_suffixes(ext_param_config_type *, int,
                                        stringlist_type *);
// FieldConfig
void *field_config_alloc_empty(const char *, ecl_grid_type *, void *, bool);
void field_config_free(field_config_type *);
field_type_enum field_config_get_type(field_config_type *);
int field_config_get_truncation_mode(field_config_type *);
double field_config_get_truncation_min(field_config_type *);
double field_config_get_truncation_max(field_config_type *);
const char *field_config_get_init_transform_name(field_config_type *);
const char *field_config_get_output_transform_name(field_config_type *);
bool field_config_ijk_active(field_config_type *, int, int, int);
int field_config_get_nx(field_config_type *);
int field_config_get_ny(field_config_type *);
int field_config_get_nz(field_config_type *);
ecl_grid_type *field_config_get_grid(field_config_type *);
int field_config_get_data_size_from_grid(field_config_type *);
enkf_field_file_format_enum field_config_default_export_format(const char *);
enkf_field_file_format_enum field_config_guess_file_type(const char *);
// GenDataConfig
void *gen_data_config_alloc_GEN_DATA_result(const char *,
                                            gen_data_file_format_type);
void gen_data_config_free(gen_data_config_type *);
gen_data_file_format_type
gen_data_config_get_output_format(gen_data_config_type *);
gen_data_file_format_type
gen_data_config_get_input_format(gen_data_config_type *);
const char *gen_data_config_get_template_file(gen_data_config_type *);
const char *gen_data_config_get_template_key(gen_data_config_type *);
int gen_data_config_get_initial_size(gen_data_config_type *);
bool gen_data_config_has_report_step(gen_data_config_type *, int);
int gen_data_config_get_data_size__(gen_data_config_type *, int);
const char *gen_data_config_get_key(gen_data_config_type *);
bool_vector_type *gen_data_config_get_active_mask(gen_data_config_type *);
int gen_data_config_num_report_step(gen_data_config_type *);
int gen_data_config_iget_report_step(gen_data_config_type *, int);
// GenKwConfig
void gen_kw_config_free(gen_kw_config_type *);
void *gen_kw_config_alloc_empty(const char *, const char *);
const char *gen_kw_config_get_template_file(gen_kw_config_type *);
void gen_kw_config_set_template_file(gen_kw_config_type *, const char *);
const char *gen_kw_config_get_parameter_file(gen_kw_config_type *);
void gen_kw_config_set_parameter_file(gen_kw_config_type *, const char *);
stringlist_type *gen_kw_config_alloc_name_list(gen_kw_config_type *);
bool gen_kw_config_should_use_log_scale(gen_kw_config_type *, int);
const char *gen_kw_config_get_key(gen_kw_config_type *);
const char *gen_kw_config_get_tag_fmt(gen_kw_config_type *);
int gen_kw_config_get_data_size(gen_kw_config_type *);
const char *gen_kw_config_iget_name(gen_kw_config_type *, int);
const char *gen_kw_config_iget_function_type(gen_kw_config_type *, int);
stringlist_type *
gen_kw_config_iget_function_parameter_names(gen_kw_config_type *, int);
double_vector_type *
gen_kw_config_iget_function_parameter_values(gen_kw_config_type *, int);
// SummaryConfig
void *summary_config_alloc(const char *, load_fail_type);
void summary_config_free(summary_config_type *);
const char *summary_config_get_var(summary_config_type *);
// Field
void field_free(field_type *);
int field_get_size(field_type *);
double field_ijk_get_double(field_type *, int, int, int);
double field_iget_double(field_type *, int);
void field_export(field_type *, const char *, fortio_type *,
                  enkf_field_file_format_enum, bool, const char *);
// EnkfNode
void *enkf_node_alloc(enkf_config_node_type *);
void *enkf_node_alloc_private_container(enkf_config_node_type *);
void enkf_node_free(enkf_node_type *);
const char *enkf_node_get_key(enkf_node_type *);
void *enkf_node_value_ptr(enkf_node_type *);
bool enkf_node_try_load(enkf_node_type *, enkf_fs_type *, node_id_type *);
bool enkf_node_store(enkf_node_type *, enkf_fs_type *, node_id_type *);
ert_impl_type_enum enkf_node_get_impl_type(enkf_node_type *);
void enkf_node_ecl_write(enkf_node_type *, const char *, void *, int);
// ExtParam
void *ext_param_alloc(ext_param_config_type *);
void ext_param_free(ext_param_type *);
void ext_param_iset(ext_param_type *, int, double);
void ext_param_key_set(ext_param_type *, const char *, double);
void ext_param_key_suffix_set(ext_param_type *, const char *, const char *,
                              double);
double ext_param_iget(ext_param_type *, int);
double ext_param_key_get(ext_param_type *, const char *);
double ext_param_key_suffix_get(ext_param_type *, const char *, const char *);
void ext_param_json_export(ext_param_type *, const char *);
void *ext_param_get_config(ext_param_type *);
// GenData
void *gen_data_alloc();
void gen_data_free(gen_data_type *);
int gen_data_get_size(gen_data_type *);
double gen_data_iget_double(gen_data_type *, int);
void gen_data_export(gen_data_type *, const char *, gen_data_file_format_type,
                     fortio_type *);
void gen_data_export_data(gen_data_type *, double_vector_type *);
// GenKw
void *gen_kw_alloc(gen_kw_config_type *);
void gen_kw_free(gen_kw_config_type *);
void gen_kw_write_export_file(gen_kw_type *, const char *);
void gen_kw_ecl_write_template(gen_kw_type *, const char *);
double gen_kw_data_iget(gen_kw_type *, int, bool);
void gen_kw_data_iset(gen_kw_type *, int, double);
void gen_kw_data_set_vector(gen_kw_type *, double_vector_type *);
double gen_kw_data_get(gen_kw_type *, const char *, bool);
void gen_kw_data_set(gen_kw_type *, const char *, double);
int gen_kw_data_size(gen_kw_type *);
bool gen_kw_data_has_key(gen_kw_type *, const char *);
void gen_kw_ecl_write(gen_kw_type *, const char *, const char *, void *);
const char *gen_kw_get_name(gen_kw_type *, int);
// Summary
void *summary_alloc(summary_config_type *);
void summary_free(summary_type *);
double summary_get(summary_type *, int);
void summary_set(summary_type *, int, double);
int summary_length(summary_type *);
double summary_undefined_value();
// BlockDataConfig
// BlockObservation
void *block_obs_alloc(const char *, block_data_config_type *, ecl_grid_type *);
void block_obs_free(block_obs_type *);
int block_obs_iget_i(block_obs_type *, int);
int block_obs_iget_j(block_obs_type *, int);
int block_obs_iget_k(block_obs_type *, int);
int block_obs_get_size(block_obs_type *);
double block_obs_iget_std(block_obs_type *, int);
double block_obs_iget_std_scaling(block_obs_type *, int);
void block_obs_update_std_scale(block_obs_type *, double, active_list_type *);
double block_obs_iget_value(block_obs_type *, int);
double block_obs_iget_depth(block_obs_type *, int);
void block_obs_append_field_obs(block_obs_type *, int, int, int, double,
                                double);
void block_obs_append_summary_obs(block_obs_type *, int, int, int, double,
                                  double);
double block_obs_iget_data(block_obs_type *, void *, int, node_id_type *);
// GenObservation
void *gen_obs_alloc__(gen_data_config_type *, const char *);
void gen_obs_free(gen_obs_type *);
void gen_obs_load_observation(gen_obs_type *, const char *);
void gen_obs_set_scalar(gen_obs_type *, double, double);
double gen_obs_iget_std(gen_obs_type *, int);
double gen_obs_iget_value(gen_obs_type *, int);
double gen_obs_iget_std_scaling(gen_obs_type *, int);
int gen_obs_get_size(gen_obs_type *);
int gen_obs_get_obs_index(gen_obs_type *, int);
void gen_obs_load_data_index(gen_obs_type *, const char *);
void gen_obs_attach_data_index(gen_obs_type *, int_vector_type *);
void gen_obs_update_std_scale(gen_obs_type *, double, active_list_type *);
void gen_obs_load_values(gen_obs_type *, int, const double *);
void gen_obs_load_std(gen_obs_type *, int, const double *);
// ObsVector
void *obs_vector_alloc(enkf_obs_impl_type, const char *,
                       enkf_config_node_type *, int);
void obs_vector_free(obs_vector_type *);
const char *obs_vector_get_state_kw(obs_vector_type *);
const char *obs_vector_get_key(obs_vector_type *);
void *obs_vector_iget_node(obs_vector_type *, int);
int obs_vector_get_num_active(obs_vector_type *);
bool obs_vector_iget_active(obs_vector_type *, int);
enkf_obs_impl_type obs_vector_get_impl_type(obs_vector_type *);
void obs_vector_install_node(obs_vector_type *, int, void *);
int obs_vector_get_next_active_step(obs_vector_type *, int);
bool obs_vector_has_data(obs_vector_type *, bool_vector_type *, enkf_fs_type *);
enkf_config_node_type *obs_vector_get_config_node(obs_vector_type *);
double obs_vector_total_chi2(obs_vector_type *, enkf_fs_type *, int);
const char *obs_vector_get_obs_key(obs_vector_type *);
local_obsdata_node_type *obs_vector_alloc_local_node(obs_vector_type *);
// SummaryObservation
void *summary_obs_alloc(const char *, const char *, double, double,
                        const char *, double);
void summary_obs_free(summary_obs_type *);
double summary_obs_get_value(summary_obs_type *);
double summary_obs_get_std(summary_obs_type *);
double summary_obs_get_std_scaling(summary_obs_type *);
const char *summary_obs_get_summary_key(summary_obs_type *);
void summary_obs_update_std_scale(summary_obs_type *, double,
                                  active_list_type *);
void summary_obs_set_std_scale(summary_obs_type *, double);
// EnsemblePlotGenDataVector
int enkf_plot_genvector_get_size(ensemble_plot_gen_data_vector_type *);
double enkf_plot_genvector_iget(ensemble_plot_gen_data_vector_type *, int);
// EnsemblePlotGenKWVector
int enkf_plot_gen_kw_vector_get_size(ensemble_plot_gen_kw_vector_type *);
double enkf_plot_gen_kw_vector_iget(ensemble_plot_gen_kw_vector_type *, int);
// EnsemblePlotData
void *enkf_plot_data_alloc(enkf_config_node_type *);
void enkf_plot_data_load(ensemble_plot_data_type *, enkf_fs_type *,
                         const char *, bool_vector_type *);
int enkf_plot_data_get_size(ensemble_plot_data_type *);
ensemble_plot_data_vector_type *enkf_plot_data_iget(ensemble_plot_data_type *,
                                                    int);
void enkf_plot_data_free(ensemble_plot_data_type *);
// EnsemblePlotDataVector
int enkf_plot_tvector_size(ensemble_plot_data_vector_type *);
double enkf_plot_tvector_iget_value(ensemble_plot_data_vector_type *, int);
time_t enkf_plot_tvector_iget_time(ensemble_plot_data_vector_type *, int);
bool enkf_plot_tvector_iget_active(ensemble_plot_data_vector_type *, int);
// EnsemblePlotGenData
void *enkf_plot_gendata_alloc(enkf_config_node_type *);
int enkf_plot_gendata_get_size(ensemble_plot_gen_data_type *);
void enkf_plot_gendata_load(ensemble_plot_gen_data_type *, enkf_fs_type *, int,
                            bool_vector_type *);
ensemble_plot_gen_data_vector_type *
enkf_plot_gendata_iget(ensemble_plot_gen_data_type *, int);
double_vector_type *
enkf_plot_gendata_get_min_values(ensemble_plot_gen_data_type *);
double_vector_type *
enkf_plot_gendata_get_max_values(ensemble_plot_gen_data_type *);
void enkf_plot_gendata_free(ensemble_plot_gen_data_type *);
// EnsemblePlotGenKW
void *enkf_plot_gen_kw_alloc(enkf_config_node_type *);
int enkf_plot_gen_kw_get_size(ensemble_plot_gen_kw_type *);
void enkf_plot_gen_kw_load(ensemble_plot_gen_kw_type *, enkf_fs_type *, bool,
                           int, bool_vector_type *);
ensemble_plot_gen_kw_vector_type *
enkf_plot_gen_kw_iget(ensemble_plot_gen_kw_type *, int);
const char *enkf_plot_gen_kw_iget_key(ensemble_plot_gen_kw_type *, int);
int enkf_plot_gen_kw_get_keyword_count(ensemble_plot_gen_kw_type *);
bool enkf_plot_gen_kw_should_use_log_scale(ensemble_plot_gen_kw_type *, int);
void enkf_plot_gen_kw_free(ensemble_plot_gen_kw_type *);
// TimeMap
void *time_map_alloc();
bool time_map_fread(time_map_type *, const char *);
void time_map_fwrite(time_map_type *, const char *);
bool time_map_fscanf(time_map_type *, const char *);
double time_map_iget_sim_days(time_map_type *, int);
time_t time_map_iget(time_map_type *, int);
int time_map_get_size(time_map_type *);
bool time_map_try_update(time_map_type *, int, time_t);
bool time_map_is_strict(time_map_type *);
void time_map_set_strict(time_map_type *, bool);
int time_map_lookup_time(time_map_type *, time_t);
int time_map_lookup_time_with_tolerance(time_map_type *, time_t, int, int);
int time_map_lookup_days(time_map_type *, double);
int time_map_get_last_step(time_map_type *);
void time_map_summary_upgrade107(time_map_type *, ecl_sum_type *);
void time_map_free(time_map_type *);
// Driver
void *queue_driver_alloc(queue_driver_enum);
void queue_driver_free(driver_type *);
void queue_driver_set_option(driver_type *, const char *, const char *);
const char *queue_driver_get_option(driver_type *, const char *);
void queue_driver_free_job(driver_type *, job_type *);
job_status_type_enum queue_driver_get_status(driver_type *, job_type *);
void queue_driver_kill_job(driver_type *, job_type *);
int queue_driver_get_max_running(driver_type *);
void queue_driver_set_max_running(driver_type *, int);
const char *queue_driver_get_name(driver_type *);
// EnvironmentVarlist
void *env_varlist_alloc();
void env_varlist_free(env_varlist_type *);
void env_varlist_setenv(env_varlist_type *, const char *, const char *);
int env_varlist_get_size(env_varlist_type *);
// ExtJob
void *ext_job_fscanf_alloc(const char *, const char *, bool, const char *,
                           bool);
void ext_job_free(ext_job_type *);
const char *ext_job_get_help_text(ext_job_type *);
const char *ext_job_get_name(ext_job_type *);
void ext_job_set_private_args_from_string(ext_job_type *, const char *);
int ext_job_is_private(ext_job_type *);
const char *ext_job_get_config_file(ext_job_type *);
void ext_job_set_config_file(ext_job_type *, const char *);
const char *ext_job_get_stdin_file(ext_job_type *);
void ext_job_set_stdin_file(ext_job_type *, const char *);
const char *ext_job_get_stdout_file(ext_job_type *);
void ext_job_set_stdout_file(ext_job_type *, const char *);
const char *ext_job_get_stderr_file(ext_job_type *);
void ext_job_set_stderr_file(ext_job_type *, const char *);
const char *ext_job_get_target_file(ext_job_type *);
void ext_job_set_target_file(ext_job_type *, const char *);
const char *ext_job_get_executable(ext_job_type *);
void ext_job_set_executable(ext_job_type *, const char *);
const char *ext_job_get_error_file(ext_job_type *);
const char *ext_job_get_start_file(ext_job_type *);
int ext_job_get_max_running(ext_job_type *);
void ext_job_set_max_running(ext_job_type *, int);
int ext_job_get_max_running_minutes(ext_job_type *);
void ext_job_set_max_running_minutes(ext_job_type *, int);
int ext_job_get_min_arg(ext_job_type *);
int ext_job_get_max_arg(ext_job_type *);
config_content_type_enum ext_job_iget_argtype(ext_job_type *, int);
string_hash_type *ext_job_get_environment(ext_job_type *);
void ext_job_add_environment(ext_job_type *, const char *, const char *);
const char *ext_job_get_license_path(ext_job_type *);
stringlist_type *ext_job_get_arglist(ext_job_type *);
void ext_job_set_args(ext_job_type *, stringlist_type *);
stringlist_type *ext_job_get_argvalues(ext_job_type *);
void ext_job_clear_environment(ext_job_type *);
void ext_job_save(ext_job_type *);
// ExtJoblist
void *ext_joblist_alloc();
void ext_joblist_free(ext_joblist_type *);
stringlist_type *ext_joblist_alloc_list(ext_joblist_type *);
ext_job_type *ext_joblist_get_job(ext_joblist_type *, const char *);
int ext_joblist_del_job(ext_joblist_type *, const char *);
int ext_joblist_has_job(ext_joblist_type *, const char *);
void ext_joblist_add_job(ext_joblist_type *, const char *, ext_job_type *);
hash_type *ext_joblist_get_jobs(ext_joblist_type *);
int ext_joblist_get_size(ext_joblist_type *);
// ForwardModel
void *forward_model_alloc(ext_joblist_type *);
void forward_model_free(forward_model_type *);
void forward_model_clear(forward_model_type *);
ext_job_type *forward_model_add_job(forward_model_type *, const char *);
stringlist_type *forward_model_alloc_joblist(forward_model_type *);
ext_job_type *forward_model_iget_job(forward_model_type *, int);
int forward_model_get_length(forward_model_type *);
void forward_model_formatted_fprintf(forward_model_type *, const char *,
                                     const char *, const char *,
                                     subst_list_type *, int,
                                     env_varlist_type *);
// Job
// JobQueueNode
void *job_queue_node_alloc_python(const char *, const char *, const char *, int,
                                  stringlist_type *, int, const char *,
                                  const char *, const char *);
void job_queue_node_free(job_queue_node_type *);
job_submit_status_type_enum job_queue_node_submit_simple(job_queue_node_type *,
                                                         driver_type *);
bool job_queue_node_kill_simple(job_queue_node_type *, driver_type *);
job_status_type_enum job_queue_node_get_status(job_queue_node_type *);
bool job_queue_node_update_status_simple(job_queue_node_type *, driver_type *);
void job_queue_node_set_status(job_queue_node_type *, job_status_type_enum);
int job_queue_node_get_submit_attempt(job_queue_node_type *);
// JobQueue
void *job_queue_alloc(int, const char *, const char *, const char *);
bool job_queue_start_user_exit(job_queue_type *);
bool job_queue_get_user_exit(job_queue_type *);
void job_queue_free(job_queue_type *);
void job_queue_set_max_job_duration(job_queue_type *, int);
int job_queue_get_max_job_duration(job_queue_type *);
void job_queue_set_driver(job_queue_type *, void *);
bool job_queue_kill_job(job_queue_type *, int);
void job_queue_run_jobs_threaded(job_queue_type *, int, bool);
void *job_queue_iget_driver_data(job_queue_type *, int);
int job_queue_get_num_running(job_queue_type *);
int job_queue_get_num_complete(job_queue_type *);
int job_queue_get_num_waiting(job_queue_type *);
int job_queue_get_num_pending(job_queue_type *);
bool job_queue_is_running(job_queue_type *);
void job_queue_submit_complete(job_queue_type *);
time_t job_queue_iget_sim_start(job_queue_type *, int);
int job_queue_get_active_size(job_queue_type *);
void job_queue_set_pause_on(job_queue_type *);
void job_queue_set_pause_off(job_queue_type *);
int job_queue_get_max_submit(job_queue_type *);
job_status_type_enum job_queue_iget_job_status(job_queue_type *, int);
const char *job_queue_get_ok_file(job_queue_type *);
const char *job_queue_get_exit_file(job_queue_type *);
const char *job_queue_get_status_file(job_queue_type *);
int job_queue_add_job_node(job_queue_type *, job_queue_node_type *);
// Workflow
void *workflow_alloc(const char *, workflow_joblist_type *);
void workflow_free(workflow_type *);
int workflow_size(workflow_type *);
workflow_job_type *workflow_iget_job(workflow_type *, int);
stringlist_type *workflow_iget_arguments(workflow_type *, int);
bool workflow_try_compile(workflow_type *, subst_list_type *);
config_error_type *workflow_get_last_error(workflow_type *);
const char *worflow_get_src_file(workflow_type *);
// WorkflowJob
void *workflow_job_alloc(const char *, bool);
config_parser_type *workflow_job_alloc_config();
workflow_job_type *workflow_job_config_alloc(const char *, config_parser_type *,
                                             const char *);
void workflow_job_free(workflow_job_type *);
const char *workflow_job_get_name(workflow_job_type *);
bool workflow_job_internal(workflow_job_type *);
bool workflow_job_is_internal_script(workflow_job_type *);
const char *workflow_job_get_internal_script_path(workflow_job_type *);
const char *workflow_job_get_function(workflow_job_type *);
const char *workflow_job_get_module(workflow_job_type *);
const char *workflow_job_get_executable(workflow_job_type *);
int workflow_job_get_min_arg(workflow_job_type *);
int workflow_job_get_max_arg(workflow_job_type *);
config_content_type_enum workflow_job_iget_argtype(workflow_job_type *, int);
// WorkflowJoblist
void *workflow_joblist_alloc();
void workflow_joblist_free(workflow_joblist_type *);
void workflow_joblist_add_job(workflow_joblist_type *, workflow_job_type *);
bool workflow_joblist_add_job_from_file(workflow_joblist_type *, const char *,
                                        const char *);
bool workflow_joblist_has_job(workflow_joblist_type *, const char *);
workflow_job_type *workflow_joblist_get_job(workflow_joblist_type *,
                                            const char *);
// History
void *history_alloc_from_refcase(ecl_sum_type *, bool);
const char *history_get_source_string(history_source_enum);
void history_free(history_type *);
// Matrix
void *matrix_alloc(int, int);
matrix_type *matrix_alloc_identity(int);
matrix_type *matrix_alloc_transpose(matrix_type *);
void matrix_inplace_transpose(matrix_type *);
matrix_type *matrix_alloc_copy(matrix_type *);
matrix_type *matrix_alloc_sub_copy(matrix_type *, int, int, int, int);
void matrix_free(matrix_type *);
double matrix_iget(matrix_type *, int, int);
void matrix_iset(matrix_type *, int, int, double);
void matrix_scalar_set(matrix_type *, double);
void matrix_scale_column(matrix_type *, int, double);
void matrix_scale_row(matrix_type *, int, double);
void matrix_copy_column(matrix_type *, matrix_type *, int, int);
int matrix_get_rows(matrix_type *);
int matrix_get_columns(matrix_type *);
bool matrix_equal(matrix_type *, matrix_type *);
void matrix_pretty_print(matrix_type *, const char *, const char *);
void matrix_fprintf(matrix_type *, const char *, FILE *);
void matrix_random_init(matrix_type *, rng_type *);
void matrix_dump_csv(matrix_type *, const char *);
matrix_type *matrix_alloc_matmul(matrix_type *, matrix_type *);
// PathFormat
void *path_fmt_alloc_directory_fmt(const char *);
const char *path_fmt_get_fmt(path_fmt_type *);
void path_fmt_free(path_fmt_type *);
// SubstitutionList
void *subst_list_alloc(void *);
void subst_list_free(subst_list_type *);
int subst_list_get_size(subst_list_type *);
const char *subst_list_iget_key(subst_list_type *, int);
const char *subst_list_get_value(subst_list_type *, const char *);
bool subst_list_has_key(subst_list_type *, const char *);
const char *subst_list_get_doc_string(subst_list_type *, const char *);
void subst_list_append_copy(subst_list_type *, const char *, const char *,
                            const char *);
// UIReturn
void *ui_return_alloc(ui_return_status);
void ui_return_free(ui_return_type *);
ui_return_status ui_return_get_status(ui_return_type *);
const char *ui_return_get_help(ui_return_type *);
bool ui_return_add_help(ui_return_type *, const char *);
bool ui_return_add_error(ui_return_type *, const char *);
int ui_return_get_error_count(ui_return_type *);
const char *ui_return_get_last_error(ui_return_type *);
const char *ui_return_get_first_error(ui_return_type *);
const char *ui_return_iget_error(ui_return_type *, int);
}
