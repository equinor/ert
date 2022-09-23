/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'res_config.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <ert/config/config_settings.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("res_config", m) {
    using namespace py::literals;
    m.def(
        "init_config_parser",
        [](py::object py_config_parser) {
            auto config_parser =
                ert::from_cwrap<config_parser_type>(py_config_parser);
            ert_workflow_list_add_config_items(config_parser);
            analysis_config_add_config_items(config_parser);
            ensemble_config_add_config_items(config_parser);
            auto item =
                config_add_schema_item(config_parser, ECLBASE_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);

            item = config_add_schema_item(config_parser, DATA_FILE_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);

            item = config_add_schema_item(config_parser, REFCASE_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_PATH);

            item = config_add_schema_item(config_parser, GRID_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_EXISTING_PATH);
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

            stringlist_type *refcase_dep = stringlist_alloc_new();
            stringlist_append_copy(refcase_dep, REFCASE_KEY);
            config_schema_item_set_required_children_on_value(
                item, "REFCASE_SIMULATED", refcase_dep);
            config_schema_item_set_required_children_on_value(
                item, "REFCASE_HISTORY", refcase_dep);

            stringlist_free(refcase_dep);

            item =
                config_add_schema_item(config_parser, RUNPATH_FILE_KEY, false);
            config_schema_item_set_argc_minmax(item, 1, 1);
            config_schema_item_iset_type(item, 0, CONFIG_PATH);

            item =
                config_add_schema_item(config_parser, HOOK_WORKFLOW_KEY, false);
            config_schema_item_set_argc_minmax(item, 2, 2);
            config_schema_item_iset_type(item, 0, CONFIG_STRING);
            config_schema_item_iset_type(item, 1, CONFIG_STRING);
            {
                stringlist_type *argv = stringlist_alloc_new();

                stringlist_append_copy(argv, RUN_MODE_PRE_SIMULATION_NAME);
                stringlist_append_copy(argv, RUN_MODE_POST_SIMULATION_NAME);
                stringlist_append_copy(argv, RUN_MODE_PRE_UPDATE_NAME);
                stringlist_append_copy(argv, RUN_MODE_POST_UPDATE_NAME);
                config_schema_item_set_indexed_selection_set(item, 1, argv);

                stringlist_free(argv);
            }
            site_config_add_config_items(config_parser, false);
            config_add_key_value(config_parser, RES_CONFIG_FILE_KEY, false,
                                 CONFIG_EXISTING_PATH);
            config_add_key_value(config_parser, CONFIG_DIRECTORY_KEY, false,
                                 CONFIG_EXISTING_PATH);
        },
        py::arg("config_parser"));
}
