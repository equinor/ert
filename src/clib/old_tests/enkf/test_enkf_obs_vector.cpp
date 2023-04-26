#include <vector>

#include <ert/util/test_util.h>
#include <ert/util/type_vector_functions.h>

#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/summary_obs.hpp>

void test_create(enkf_config_node_type *config_node) {
    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "OBS", config_node, 100);
    {

        {
            summary_obs_type *obs_node =
                summary_obs_alloc("FOPT", "FOPT", 10, 1);
            obs_vector_install_node(obs_vector, 10, obs_node);
            const std::vector<int> step_list =
                obs_vector_get_step_list(obs_vector);
            test_assert_int_equal(1, step_list.size());
            test_assert_int_equal(10, step_list[0]);
        }

        {
            summary_obs_type *obs_node =
                summary_obs_alloc("FOPT", "FOPT", 10, 1);
            obs_vector_install_node(obs_vector, 10, obs_node);
            const std::vector<int> step_list =
                obs_vector_get_step_list(obs_vector);
            test_assert_int_equal(1, step_list.size());
            test_assert_int_equal(10, step_list[0]);
        }

        {
            summary_obs_type *obs_node =
                summary_obs_alloc("FOPT", "FOPT", 10, 1);
            obs_vector_install_node(obs_vector, 5, obs_node);
            const std::vector<int> step_list =
                obs_vector_get_step_list(obs_vector);
            test_assert_int_equal(2, step_list.size());
            test_assert_int_equal(5, step_list[0]);
            test_assert_int_equal(10, step_list[1]);
        }

        {
            summary_obs_type *obs_node =
                summary_obs_alloc("FOPT", "FOPT", 10, 1);
            obs_vector_install_node(obs_vector, 15, obs_node);
            const std::vector<int> step_list =
                obs_vector_get_step_list(obs_vector);
            test_assert_int_equal(3, step_list.size());
            test_assert_int_equal(5, step_list[0]);
            test_assert_int_equal(10, step_list[1]);
            test_assert_int_equal(15, step_list[2]);
        }
    }
    obs_vector_free(obs_vector);
}

int main(int argc, char **argv) {
    enkf_config_node_type *config_node = enkf_config_node_alloc_summary("FOPR");
    { test_create(config_node); }
    enkf_config_node_free(config_node);
}
