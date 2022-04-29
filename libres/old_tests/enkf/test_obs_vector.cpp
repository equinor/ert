/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_obs_vector_tests.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <vector>

#include <ert/enkf/block_obs.hpp>
#include <ert/enkf/gen_obs.hpp>
#include <ert/enkf/obs_vector.hpp>
#include <ert/enkf/summary_obs.hpp>
#include <ert/util/test_util.h>

bool alloc_strippedparameters_noerrors() {
    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 0);
    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_summary_nodata_no_errors() {
    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 0);
    auto local_node = obs_vector_alloc_local_node(obs_vector);
    obs_vector_scale_std(obs_vector, local_node, 2.0);
    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_summarysingleobservation_no_errors() {
    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 1);
    summary_obs_type *summary_obs =
        summary_obs_alloc("SummaryKey", "ObservationKey", 43.2, 2.0);
    obs_vector_install_node(obs_vector, 0, summary_obs);
    test_assert_double_equal(2.0, summary_obs_get_std(summary_obs));
    test_assert_double_equal(1.0, summary_obs_get_std_scaling(summary_obs));

    {
        auto local_node = obs_vector_alloc_local_node(obs_vector);
        obs_vector_scale_std(obs_vector, local_node, 2.0);
    }
    test_assert_double_equal(2.0, summary_obs_get_std_scaling(summary_obs));

    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_summarymanyobservations_no_errors() {
    int num_observations = 100;
    double scaling_factor = 1.456;

    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, num_observations);

    test_assert_bool_equal(0, obs_vector_get_num_active(obs_vector));

    summary_obs_type *observations[num_observations];
    for (int i = 0; i < num_observations; i++) {
        summary_obs_type *summary_obs =
            summary_obs_alloc("SummaryKey", "ObservationKey", 43.2, i);
        obs_vector_install_node(obs_vector, i, summary_obs);
        observations[i] = summary_obs;
    }

    for (int i = 0; i < num_observations; i++) {
        summary_obs_type *before_scale = observations[i];
        test_assert_double_equal(i, summary_obs_get_std(before_scale));
    }

    test_assert_bool_equal(num_observations,
                           obs_vector_get_num_active(obs_vector));
    {
        auto local_node = obs_vector_alloc_local_node(obs_vector);
        obs_vector_scale_std(obs_vector, local_node, scaling_factor);
    }
    for (int i = 0; i < num_observations; i++) {
        summary_obs_type *after_scale = observations[i];
        test_assert_double_equal(scaling_factor,
                                 summary_obs_get_std_scaling(after_scale));
    }

    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_block_nodata_no_errors() {
    obs_vector_type *obs_vector = obs_vector_alloc(BLOCK_OBS, "WHAT", NULL, 0);
    auto local_node = obs_vector_alloc_local_node(obs_vector);
    obs_vector_scale_std(obs_vector, local_node, 2.0);
    obs_vector_free(obs_vector);
    return true;
}

block_obs_type *create_block_obs(ecl_grid_type *grid, int size, double value,
                                 double std_dev) {
    field_config_type *field_config =
        field_config_alloc_empty("PRESSURE", grid, NULL, false);
    std::vector<int> i(size);
    std::vector<int> j(size);
    std::vector<int> k(size);
    std::vector<double> obs_value(size);
    std::vector<double> obs_std(size);

    for (int num = 0; num < size; num++) {
        obs_value[num] = value;
        obs_std[num] = std_dev;
        i[num] = num;
        j[num] = num;
        k[num] = num;
    }

    block_obs_type *block_obs = block_obs_alloc_complete(
        "Label", SOURCE_FIELD, NULL, field_config, grid, size, i.data(),
        j.data(), k.data(), obs_value.data(), obs_std.data());

    field_config_free(field_config);

    return block_obs;
}

bool scale_std_block100observations_no_errors() {
    int num_observations = 100;
    int num_points = 10;

    obs_vector_type *obs_vector =
        obs_vector_alloc(BLOCK_OBS, "WHAT", NULL, num_observations);
    ecl_grid_type *grid = ecl_grid_alloc_rectangular(
        num_points, num_points, num_points, 1.0, 1.0, 1.0, NULL);

    double scale_factor = 3.3;
    double obs_value = 44;
    double obs_std = 3.2;

    block_obs_type *observations[num_observations];

    for (int i = 0; i < num_observations; i++) {
        block_obs_type *block_obs =
            create_block_obs(grid, num_points, obs_value, obs_std);
        obs_vector_install_node(obs_vector, i, block_obs);
        observations[i] = block_obs;
    }

    for (int i = 0; i < num_observations; i++) {
        for (int point_nr = 0; point_nr < num_points; point_nr++) {
            double value = block_obs_iget_value(observations[i], point_nr);
            double std = block_obs_iget_std(observations[i], point_nr);
            test_assert_double_equal(obs_value, value);
            test_assert_double_equal(obs_std, std);
        }
    }

    {
        auto local_node = obs_vector_alloc_local_node(obs_vector);
        obs_vector_scale_std(obs_vector, local_node, scale_factor);
    }

    for (int i = 0; i < num_observations; i++) {
        for (int point_nr = 0; point_nr < num_points; point_nr++) {
            double value = block_obs_iget_value(observations[i], point_nr);
            double std = block_obs_iget_std(observations[i], point_nr);
            double std_scale =
                block_obs_iget_std_scaling(observations[i], point_nr);
            test_assert_double_equal(obs_value, value);
            test_assert_double_equal(obs_std, std);
            test_assert_double_equal(scale_factor, std_scale);
        }
    }

    ecl_grid_free(grid);
    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_gen_nodata_no_errors() {
    obs_vector_type *obs_vector = obs_vector_alloc(GEN_OBS, "WHAT", NULL, 0);
    auto local_node = obs_vector_alloc_local_node(obs_vector);
    obs_vector_scale_std(obs_vector, local_node, 2.0);
    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_gen_withdata_no_errors() {
    int num_observations = 100;
    double value = 42;
    double std_dev = 2.2;
    double multiplier = 3.4;

    obs_vector_type *obs_vector =
        obs_vector_alloc(GEN_OBS, "WHAT", NULL, num_observations);

    gen_obs_type *observations[num_observations];
    for (int i = 0; i < num_observations; i++) {
        gen_obs_type *gen_obs =
            gen_obs_alloc(NULL, "WWCT-GEN", NULL, value, std_dev, NULL, NULL);
        obs_vector_install_node(obs_vector, i, gen_obs);
        observations[i] = gen_obs;
    }

    {
        auto local_node = obs_vector_alloc_local_node(obs_vector);
        obs_vector_scale_std(obs_vector, local_node, multiplier);
    }

    for (int i = 0; i < num_observations; i++) {
        char *index_key = util_alloc_sprintf("%d", 0);
        double value_new, std_new;
        bool valid;
        gen_obs_user_get_with_data_index(observations[i], index_key, &value_new,
                                         &std_new, &valid);
        test_assert_double_equal(std_dev, std_new);
        test_assert_double_equal(value, value_new);
        test_assert_double_equal(multiplier,
                                 gen_obs_iget_std_scaling(observations[i], 0));
        free(index_key);
    }

    obs_vector_free(obs_vector);
    return true;
}

int main(int argc, char **argv) {
    test_assert_bool_equal(alloc_strippedparameters_noerrors(), true);
    test_assert_bool_equal(scale_std_summary_nodata_no_errors(), true);
    test_assert_bool_equal(scale_std_summarysingleobservation_no_errors(),
                           true);
    test_assert_bool_equal(scale_std_summarymanyobservations_no_errors(), true);

    test_assert_bool_equal(scale_std_block_nodata_no_errors(), true);
    test_assert_bool_equal(scale_std_block100observations_no_errors(), true);

    test_assert_bool_equal(scale_std_gen_nodata_no_errors(), true);
    test_assert_bool_equal(scale_std_gen_withdata_no_errors(), true);

    exit(0);
}
