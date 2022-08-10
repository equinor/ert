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

bool scale_std_summarysingleobservation_no_errors() {
    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "WHAT", NULL, 1);
    summary_obs_type *summary_obs =
        summary_obs_alloc("SummaryKey", "ObservationKey", 43.2, 2.0);
    obs_vector_install_node(obs_vector, 0, summary_obs);
    test_assert_double_equal(2.0, summary_obs_get_std(summary_obs));
    test_assert_double_equal(1.0, summary_obs_get_std_scaling(summary_obs));

    ActiveList active_list;
    summary_obs_update_std_scale(summary_obs, 2.0, &active_list);
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

    ActiveList active_list;
    for (int i = 0; i < num_observations; i++)
        summary_obs_update_std_scale(observations[i], scaling_factor,
                                     &active_list);

    for (int i = 0; i < num_observations; i++) {
        summary_obs_type *after_scale = observations[i];
        test_assert_double_equal(scaling_factor,
                                 summary_obs_get_std_scaling(after_scale));
    }

    obs_vector_free(obs_vector);
    return true;
}

bool scale_std_gen_nodata_no_errors() {
    obs_vector_type *obs_vector = obs_vector_alloc(GEN_OBS, "WHAT", NULL, 0);
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

    ActiveList active_list;
    for (int i = 0; i < num_observations; i++)
        gen_obs_update_std_scale(observations[i], multiplier, &active_list);

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
    test_assert_bool_equal(scale_std_summarysingleobservation_no_errors(),
                           true);
    test_assert_bool_equal(scale_std_summarymanyobservations_no_errors(), true);

    test_assert_bool_equal(scale_std_gen_nodata_no_errors(), true);
    test_assert_bool_equal(scale_std_gen_withdata_no_errors(), true);

    exit(0);
}
