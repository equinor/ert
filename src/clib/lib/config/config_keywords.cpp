#include <ert/config/config_settings.hpp>

#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/python.hpp>

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

static void add_jobname_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, JOBNAME_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 1);
}

static void add_forward_model_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, FORWARD_MODEL_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);
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

static void add_obs_config_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, OBS_CONFIG_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 1);
    config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);
}

static void add_gen_kw_export_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, GEN_KW_EXPORT_NAME_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 1);
}

static void add_history_source_keyword(config_parser_type *config_parser) {
    auto item =
        config_add_schema_item(config_parser, HISTORY_SOURCE_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 1);

    stringlist_type *argv = stringlist_alloc_new();
    stringlist_append_copy(argv, "SCHEDULE");
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

static void add_runpath_file_keyword(config_parser_type *config_parser) {
    auto item = config_add_schema_item(config_parser, RUNPATH_FILE_KEY, false);
    config_schema_item_set_argc_minmax(item, 1, 1);
    config_schema_item_iset_type(item, 0, CONFIG_PATH);
}

ERT_CLIB_SUBMODULE("config_keywords", m) {
    using namespace py::literals;
    m.def(
        "init_res_config_parser",
        [](py::object py_config_parser) {
            auto config_parser =
                ert::from_cwrap<config_parser_type>(py_config_parser);
            ert_workflow_list_add_config_items(config_parser);
            config_schema_item_type *item;

            config_add_key_value(config_parser, ENKF_ALPHA_KEY, false,
                                 CONFIG_FLOAT);
            config_add_key_value(config_parser, STD_CUTOFF_KEY, false,
                                 CONFIG_FLOAT);
            config_settings_init_parser__(UPDATE_SETTING_KEY, config_parser,
                                          false);

            config_add_key_value(config_parser, SINGLE_NODE_UPDATE_KEY, false,
                                 CONFIG_BOOL);

            config_add_key_value(config_parser, ENKF_RERUN_KEY, false,
                                 CONFIG_BOOL);
            config_add_key_value(config_parser, RERUN_START_KEY, false,
                                 CONFIG_INT);
            config_add_key_value(config_parser, UPDATE_LOG_PATH_KEY, false,
                                 CONFIG_STRING);
            config_add_key_value(config_parser, MIN_REALIZATIONS_KEY, false,
                                 CONFIG_STRING);
            config_add_key_value(config_parser, MAX_RUNTIME_KEY, false,
                                 CONFIG_INT);

            item = config_add_key_value(config_parser, STOP_LONG_RUNNING_KEY,
                                        false, CONFIG_BOOL);
            stringlist_type *child_list = stringlist_alloc_new();
            stringlist_append_copy(child_list, MIN_REALIZATIONS_KEY);
            config_schema_item_set_required_children_on_value(item, "TRUE",
                                                              child_list);
            stringlist_free(child_list);

            config_add_key_value(config_parser, ANALYSIS_SELECT_KEY, false,
                                 CONFIG_STRING);

            item =
                config_add_schema_item(config_parser, ANALYSIS_COPY_KEY, false);
            config_schema_item_set_argc_minmax(item, 2, 2);

            item = config_add_schema_item(config_parser, ANALYSIS_SET_VAR_KEY,
                                          false);
            config_schema_item_set_argc_minmax(item, 3, CONFIG_DEFAULT_ARG_MAX);
            analysis_iter_config_add_config_items(config_parser);
            ensemble_config_add_config_items(config_parser);
            ecl_config_add_config_items(config_parser);
            config_add_key_value(config_parser, RANDOM_SEED_KEY, false,
                                 CONFIG_STRING);
            config_add_key_value(config_parser, MAX_RESAMPLE_KEY, false,
                                 CONFIG_INT);
            add_num_realizations_keyword(config_parser);
            add_run_template_keyword(config_parser);
            config_add_key_value(config_parser, RUNPATH_KEY, false,
                                 CONFIG_PATH);
            config_add_key_value(config_parser, DATA_ROOT_KEY, false,
                                 CONFIG_PATH);
            config_add_key_value(config_parser, ENSPATH_KEY, false,
                                 CONFIG_PATH);
            add_jobname_keyword(config_parser);
            add_forward_model_keyword(config_parser);
            add_simulation_job_keyword(config_parser);
            add_data_kw_keyword(config_parser);
            add_obs_config_keyword(config_parser);
            config_add_key_value(config_parser, TIME_MAP_KEY, false,
                                 CONFIG_EXISTING_PATH);
            add_gen_kw_export_keyword(config_parser);
            add_history_source_keyword(config_parser);
            add_runpath_file_keyword(config_parser);
            hook_manager_add_config_items(config_parser);
            site_config_add_config_items(config_parser, false);
            config_add_key_value(config_parser, RES_CONFIG_FILE_KEY, false,
                                 CONFIG_EXISTING_PATH);
            config_add_key_value(config_parser, CONFIG_DIRECTORY_KEY, false,
                                 CONFIG_EXISTING_PATH);
        },
        py::arg("config_parser"));
}
