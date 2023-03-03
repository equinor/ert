#include <ert/config/config_parser.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/python.hpp>

static void add_float_keyword(config_parser_type *config_parser,
                              std::string keyword) {
    config_add_key_value(config_parser, keyword.c_str(), false, CONFIG_FLOAT);
}

static void add_int_keyword(config_parser_type *config_parser,
                            std::string keyword) {
    config_add_key_value(config_parser, keyword.c_str(), false, CONFIG_INT);
}

static void add_string_keyword(config_parser_type *config_parser,
                               std::string keyword) {
    config_add_key_value(config_parser, keyword.c_str(), false, CONFIG_STRING);
}

static void add_bool_keyword(config_parser_type *config_parser,
                             std::string keyword) {
    config_add_key_value(config_parser, keyword.c_str(), false, CONFIG_BOOL);
}

static void add_path_keyword(config_parser_type *config_parser,
                             std::string keyword) {
    config_add_key_value(config_parser, keyword.c_str(), false, CONFIG_PATH);
}

static void add_existing_path_keyword(config_parser_type *config_parser,
                                      std::string keyword) {
    config_add_key_value(config_parser, keyword.c_str(), false,
                         CONFIG_EXISTING_PATH);
}
static config_schema_item_type *
add_single_arg_keyword(config_parser_type *config_parser, std::string keyword) {
    auto item = config_add_schema_item(config_parser, keyword.c_str(), false);
    config_schema_item_set_argc_minmax(item, 1, 1);
    return item;
}

static void add_num_realizations_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, NUM_REALIZATIONS_KEY, true);
    config_schema_item_set_argc_minmax(item, 1, 1);
    config_schema_item_iset_type(item, 0, CONFIG_INT);
    config_add_alias(config_parser, NUM_REALIZATIONS_KEY, "NUM_REALISATIONS");
}

static void add_run_template_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, RUN_TEMPLATE_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, CONFIG_DEFAULT_ARG_MAX);
    config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);
}

static void add_forward_model_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, FORWARD_MODEL_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
    config_schema_item_disable_substitutions(item);
}

static void add_simulation_job_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, SIMULATION_JOB_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
}

static void add_data_kw_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, DATA_KW_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
}

static void add_history_source_keyword(config_parser_type *config_parser) {
    auto item = add_single_arg_keyword(config_parser, HISTORY_SOURCE_KEY);

    stringlist_type *argv = stringlist_alloc_new();
    stringlist_append_copy(argv, "REFCASE_SIMULATED");
    stringlist_append_copy(argv, "REFCASE_HISTORY");

    config_schema_item_set_common_selection_set(item, argv);
    stringlist_free(argv);

    stringlist_type *refcase_dep = stringlist_alloc_new();
    stringlist_append_copy(refcase_dep, REFCASE_KEY);
    config_schema_item_set_required_children_on_value(item, "REFCASE_SIMULATED",
                                                      refcase_dep);
    config_schema_item_set_required_children_on_value(item, "REFCASE_HISTORY",
                                                      refcase_dep);
    stringlist_free(refcase_dep);
}

static void add_stop_long_running_keyword(config_parser_type *config_parser) {
    auto item = config_add_key_value(config_parser, STOP_LONG_RUNNING_KEY,
                                     false, CONFIG_BOOL);
    stringlist_type *child_list = stringlist_alloc_new();
    stringlist_append_copy(child_list, MIN_REALIZATIONS_KEY);
    config_schema_item_set_required_children_on_value(item, "TRUE", child_list);
    stringlist_free(child_list);
}

static void add_analysis_copy_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, ANALYSIS_COPY_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
}

static void add_update_setting_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, UPDATE_SETTING_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
}

static void add_analysis_set_var_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, ANALYSIS_SET_VAR_KEY, false);
    config_schema_item_set_argc_minmax(item, 3, CONFIG_DEFAULT_ARG_MAX);
}

static void add_hook_workflow_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, HOOK_WORKFLOW_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
    config_schema_item_iset_type(item, 0, CONFIG_STRING);
    config_schema_item_iset_type(item, 1, CONFIG_STRING);

    stringlist_type *argv = stringlist_alloc_new();
    stringlist_append_copy(argv, RUN_MODE_PRE_SIMULATION_NAME);
    stringlist_append_copy(argv, RUN_MODE_POST_SIMULATION_NAME);
    stringlist_append_copy(argv, RUN_MODE_PRE_UPDATE_NAME);
    stringlist_append_copy(argv, RUN_MODE_PRE_FIRST_UPDATE_NAME);
    stringlist_append_copy(argv, RUN_MODE_POST_UPDATE_NAME);
    config_schema_item_set_indexed_selection_set(item, 1, argv);
    stringlist_free(argv);
}

static void add_set_env_keyword(config_parser_type *config_parser) {
    // You can set environment variables which will be applied to the run-time
    // environment. Can unfortunately not use constructions like
    // PATH=$PATH:/some/new/path, use the UPDATE_PATH function instead.
    auto item = config_add_schema_item(config_parser, SETENV_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
    // Do not expand $VAR expressions (that is done in util_interp_setenv()).
    config_schema_item_set_envvar_expansion(item, false);
}

static void add_update_path_keyword(config_parser_type *config_parser) {
    // UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib
    // Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
    auto item = config_add_schema_item(config_parser, UPDATE_PATH_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
    // Do not expand $VAR expressions (that is done in util_interp_setenv()).
    config_schema_item_set_envvar_expansion(item, false);
}

static void add_install_job_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, INSTALL_JOB_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, 2);
    config_schema_item_iset_type(item, 1, CONFIG_EXISTING_PATH);
}

static void add_load_workflow_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, LOAD_WORKFLOW_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 2);
    config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);
}

static void add_load_workflow_job_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, LOAD_WORKFLOW_JOB_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 2);
    config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);
}

static void add_queue_system_keyword(config_parser_type *config_parser,
                                     bool required) {
    auto item =
        config_add_schema_item(config_parser, QUEUE_SYSTEM_KEY, required);
    config_schema_item_set_argc_minmax(item, 1, 1);
}

static void add_queue_option_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, QUEUE_OPTION_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, CONFIG_DEFAULT_ARG_MAX);
    stringlist_type *argv = stringlist_alloc_new();
    stringlist_append_copy(argv, "LSF");
    stringlist_append_copy(argv, "LOCAL");
    stringlist_append_copy(argv, "TORQUE");
    stringlist_append_copy(argv, "SLURM");
    config_schema_item_set_indexed_selection_set(item, 0, argv);
}

static void add_job_script_keyword(config_parser_type *config_parser) {
    auto item = add_single_arg_keyword(config_parser, JOB_SCRIPT_KEY);
    config_schema_item_iset_type(item, 0, CONFIG_EXECUTABLE);
}

static void add_gen_kw_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, GEN_KW_KEY, false);
    config_schema_item_set_argc_minmax(item, 4, 6);
    config_schema_item_iset_type(item, 1, CONFIG_EXISTING_PATH);
    config_schema_item_iset_type(item, 2, CONFIG_STRING);
    config_schema_item_iset_type(item, 3, CONFIG_EXISTING_PATH);
}

static void
add_schedule_prediction_file_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser,
                                       SCHEDULE_PREDICTION_FILE_KEY, false);
    /* scedhule_prediction_file   filename  <parameters:> <init_files:> */
    config_schema_item_set_argc_minmax(item, 1, 3);
    config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);
    config_schema_item_set_deprecated(
        item, "The SCHEDULE_PREDICTION_FILE config key is deprecated.");
}

static void add_summary_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, SUMMARY_KEY, false);
    /* can have several summary keys on each line. */
    config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
}

static void add_surface_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, SURFACE_KEY, false);
    config_schema_item_set_argc_minmax(item, 4, 5);
}

static void add_field_keyword(config_parser_type *config_parser) {
    // the way config info is entered for fields is unfortunate because
    // it is difficult/impossible to let the config system handle run
    // time validation of the input.

    auto item = config_add_schema_item(config_parser, FIELD_KEY, false);
    config_schema_item_set_argc_minmax(item, 2, CONFIG_DEFAULT_ARG_MAX);
    // if you are using a field - you must have a grid.
    config_schema_item_add_required_children(item, GRID_KEY);
}

static void add_gen_data_keyword(config_parser_type *config) {
    auto item = config_add_schema_item(config, GEN_DATA_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
}

void init_site_config_parser(config_parser_type *config_parser) {
    add_int_keyword(config_parser, MAX_SUBMIT_KEY);
    add_int_keyword(config_parser, NUM_CPU_KEY);
    add_queue_system_keyword(config_parser, true);
    add_queue_option_keyword(config_parser);
    add_job_script_keyword(config_parser);
    add_path_keyword(config_parser, WORKFLOW_JOB_DIRECTORY_KEY);
    add_load_workflow_keyword(config_parser);
    add_load_workflow_job_keyword(config_parser);
    add_set_env_keyword(config_parser);
    add_update_path_keyword(config_parser);
    add_install_job_keyword(config_parser);
    add_path_keyword(config_parser, INSTALL_JOB_DIRECTORY_KEY);
    add_hook_workflow_keyword(config_parser);
}

ERT_CLIB_SUBMODULE("config_keywords", m) {
    using namespace py::literals;
    m.def(
        "init_site_config_parser",
        [](Cwrap<config_parser_type> config_parser) {
            init_site_config_parser(config_parser);
        },
        py::arg("config_parser"));
    m.def(
        "init_user_config_parser",
        [](Cwrap<config_parser_type> config_parser) {
            add_path_keyword(config_parser, WORKFLOW_JOB_DIRECTORY_KEY);
            add_load_workflow_keyword(config_parser);
            add_load_workflow_job_keyword(config_parser);
            add_float_keyword(config_parser, ENKF_ALPHA_KEY);
            add_float_keyword(config_parser, STD_CUTOFF_KEY);
            add_update_setting_keyword(config_parser);
            add_string_keyword(config_parser, UPDATE_LOG_PATH_KEY);
            add_string_keyword(config_parser, MIN_REALIZATIONS_KEY);
            add_int_keyword(config_parser, MAX_RUNTIME_KEY);
            add_stop_long_running_keyword(config_parser);
            add_string_keyword(config_parser, ANALYSIS_SELECT_KEY);
            add_analysis_copy_keyword(config_parser);
            add_analysis_set_var_keyword(config_parser);
            add_string_keyword(config_parser, ITER_CASE_KEY);
            add_int_keyword(config_parser, ITER_COUNT_KEY);
            add_int_keyword(config_parser, ITER_RETRY_COUNT_KEY);
            // the two fault types are just added to the config object only to
            // be able to print suitable messages before exiting.
            add_gen_kw_keyword(config_parser);
            add_schedule_prediction_file_keyword(config_parser);
            add_string_keyword(config_parser, GEN_KW_TAG_FORMAT_KEY);
            add_gen_data_keyword(config_parser);
            add_summary_keyword(config_parser);
            add_surface_keyword(config_parser);
            add_field_keyword(config_parser);
            add_single_arg_keyword(config_parser, ECLBASE_KEY);
            add_existing_path_keyword(config_parser, DATA_FILE_KEY);
            add_existing_path_keyword(config_parser, GRID_KEY);
            add_path_keyword(config_parser, REFCASE_KEY);
            add_string_keyword(config_parser, RANDOM_SEED_KEY);
            add_num_realizations_keyword(config_parser);
            add_run_template_keyword(config_parser);
            add_path_keyword(config_parser, RUNPATH_KEY);
            add_path_keyword(config_parser, ENSPATH_KEY);
            add_single_arg_keyword(config_parser, JOBNAME_KEY);
            add_forward_model_keyword(config_parser);
            add_simulation_job_keyword(config_parser);
            add_data_kw_keyword(config_parser);
            add_existing_path_keyword(config_parser, OBS_CONFIG_KEY);
            add_existing_path_keyword(config_parser, TIME_MAP_KEY);
            add_single_arg_keyword(config_parser, GEN_KW_EXPORT_NAME_KEY);
            add_history_source_keyword(config_parser);
            add_path_keyword(config_parser, RUNPATH_FILE_KEY);
            add_int_keyword(config_parser, MAX_SUBMIT_KEY);
            add_int_keyword(config_parser, NUM_CPU_KEY);
            add_queue_system_keyword(config_parser, false);
            add_queue_option_keyword(config_parser);
            add_job_script_keyword(config_parser);
            add_load_workflow_job_keyword(config_parser);
            add_set_env_keyword(config_parser);
            add_update_path_keyword(config_parser);
            add_path_keyword(config_parser, LICENSE_PATH_KEY);
            add_install_job_keyword(config_parser);
            add_path_keyword(config_parser, INSTALL_JOB_DIRECTORY_KEY);
            add_hook_workflow_keyword(config_parser);
            add_existing_path_keyword(config_parser, CONFIG_DIRECTORY_KEY);
        },
        py::arg("config_parser"));
}
