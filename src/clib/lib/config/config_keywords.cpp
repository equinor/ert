#include <ert/config/config_settings.hpp>

#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/hook_manager.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("config_keywords", m) {
    using namespace py::literals;
    m.def(
        "init_res_config_parser",
        [](py::object py_config_parser) {
            auto config_parser =
                ert::from_cwrap<config_parser_type>(py_config_parser);
            config_schema_item_type *item;

            ert_workflow_list_add_config_items(config_parser);
            analysis_config_add_config_items(config_parser);
            ensemble_config_add_config_items(config_parser);
            ecl_config_add_config_items(config_parser);
            config_add_key_value(config_parser, RANDOM_SEED_KEY, false,
                                 CONFIG_STRING);

            config_add_key_value(config_parser, MAX_RESAMPLE_KEY, false,
                                 CONFIG_INT);

            item = config_add_schema_item(config_parser, NUM_REALIZATIONS_KEY,
                                          true);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_INT);
            config_add_alias(config_parser, NUM_REALIZATIONS_KEY,
                             "NUM_REALISATIONS");

            /* Optional keywords from the model config file */
            item =
                config_add_schema_item(config_parser, RUN_TEMPLATE_KEY, false);
            config_schema_item_set_argc_minmax(item, 2, CONFIG_DEFAULT_ARG_MAX);
            config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

            config_add_key_value(config_parser, RUNPATH_KEY, false,
                                 CONFIG_PATH);
            config_add_key_value(config_parser, DATA_ROOT_KEY, false,
                                 CONFIG_PATH);
            config_add_key_value(config_parser, ENSPATH_KEY, false,
                                 CONFIG_PATH);

            item = config_add_schema_item(config_parser, JOBNAME_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);

            item =
                config_add_schema_item(config_parser, FORWARD_MODEL_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);

            item = config_add_schema_item(config_parser, SIMULATION_JOB_KEY,
                                          false);
            config_schema_item_set_argc_minmax(item, 1, CONFIG_DEFAULT_ARG_MAX);

            item = config_add_schema_item(config_parser, DATA_KW_KEY, false);
            config_schema_item_set_argc_minmax(item, 2, 2);

            item = config_add_schema_item(config_parser, OBS_CONFIG_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

            config_add_key_value(config_parser, TIME_MAP_KEY, false,
                                 CONFIG_EXISTING_PATH);

            item = config_add_schema_item(config_parser, GEN_KW_EXPORT_NAME_KEY,
                                          false);
            config_schema_item_set_argc_minmax(item, 1, 1);

            stringlist_type *refcase_dep = stringlist_alloc_new();
            stringlist_append_copy(refcase_dep, REFCASE_KEY);
            item = config_add_schema_item(config_parser, HISTORY_SOURCE_KEY,
                                          false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            {
                stringlist_type *argv = stringlist_alloc_new();
                stringlist_append_copy(argv, "SCHEDULE");
                stringlist_append_copy(argv, "REFCASE_SIMULATED");
                stringlist_append_copy(argv, "REFCASE_HISTORY");

                config_schema_item_set_common_selection_set(item, argv);
                stringlist_free(argv);
            }
            config_schema_item_set_required_children_on_value(
                item, "REFCASE_SIMULATED", refcase_dep);
            config_schema_item_set_required_children_on_value(
                item, "REFCASE_HISTORY", refcase_dep);

            stringlist_free(refcase_dep);

            item =
                config_add_schema_item(config_parser, RUNPATH_FILE_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_PATH);

            hook_manager_add_config_items(config_parser);
            site_config_add_config_items(config_parser, false);
            config_add_key_value(config_parser, RES_CONFIG_FILE_KEY, false,
                                 CONFIG_EXISTING_PATH);
            config_add_key_value(config_parser, CONFIG_DIRECTORY_KEY, false,
                                 CONFIG_EXISTING_PATH);
        },
        py::arg("config_parser"));
}
