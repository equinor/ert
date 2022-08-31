#include <string>

#include "catch2/catch.hpp"
#include <Eigen/Dense>

#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/obs_data.hpp>

SCENARIO("Creating eigen vectors from obs_data [obs_data]") {
    GIVEN("A obs_data with one obs_block") {
        double global_std_scaling = 1.0;
        obs_data_type *obs_data = obs_data_alloc(global_std_scaling);

        const char *obs_key = "obs_block_0";
        const int obs_size = 3;

        obs_block_type *obs_block =
            obs_data_add_block(obs_data, obs_key, obs_size);
        obs_block_iset(obs_block, 0, 5.0, 0.3);
        obs_block_iset(obs_block, 2, 15.0, 0.5);

        THEN("loading as vector") {
            Eigen::VectorXd observation_errors =
                obs_data_errors_as_vector(obs_data);
            Eigen::VectorXd observation_values =
                obs_data_values_as_vector(obs_data);
            REQUIRE(observation_values == Eigen::Vector2d(5.0, 15.0));
            REQUIRE(observation_errors == Eigen::Vector2d(0.3, 0.5));
        }
    }
}
