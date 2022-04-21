/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'enkf_obs_fs.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdio.h>
#include <stdlib.h>

#include <ert/util/test_util.h>
#include <ert/util/type_vector_functions.h>

#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/ert_test_context.hpp>

void test_iget(ert_test_context_type *test_context) {
    enkf_main_type *enkf_main = ert_test_context_get_main(test_context);
    enkf_obs_type *enkf_obs = enkf_main_get_obs(enkf_main);

    test_assert_int_equal(32, enkf_obs_get_size(enkf_obs));
    for (int iobs = 0; iobs < enkf_obs_get_size(enkf_obs); iobs++) {
        obs_vector_type *vec1 = enkf_obs_iget_vector(enkf_obs, iobs);
        obs_vector_type *vec2 =
            enkf_obs_get_vector(enkf_obs, obs_vector_get_key(vec1));

        test_assert_ptr_equal(vec1, vec2);
    }
}

void test_container(ert_test_context_type *test_context) {
    enkf_config_node_type *config_node =
        enkf_config_node_new_container("CONTAINER");
    enkf_config_node_type *wwct1_node =
        enkf_config_node_alloc_summary("WWCT:OP_1", LOAD_FAIL_SILENT);
    enkf_config_node_type *wwct2_node =
        enkf_config_node_alloc_summary("WWCT:OP_2", LOAD_FAIL_SILENT);
    enkf_config_node_type *wwct3_node =
        enkf_config_node_alloc_summary("WWCT:OP_3", LOAD_FAIL_SILENT);

    enkf_config_node_update_container(config_node, wwct1_node);
    enkf_config_node_update_container(config_node, wwct2_node);
    enkf_config_node_update_container(config_node, wwct3_node);
    {
        enkf_node_type *container = enkf_node_deep_alloc(config_node);
        enkf_node_free(container);
    }

    enkf_config_node_free(wwct3_node);
    enkf_config_node_free(wwct2_node);
    enkf_config_node_free(wwct1_node);
    enkf_config_node_free(config_node);
}

int main(int argc, char **argv) {
    util_install_signals();
    {
        const char *config_file = argv[1];
        ert_test_context_type *test_context =
            ert_test_context_alloc("ENKF_OBS_FS", config_file);
        {
            test_iget(test_context);
            test_container(test_context);
        }
        ert_test_context_free(test_context);
        exit(0);
    }
}
