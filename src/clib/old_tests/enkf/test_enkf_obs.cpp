#include <ert/util/test_util.h>

#include <ert/ecl/ecl_grid.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_obs.hpp>
#include <ert/enkf/summary_obs.hpp>

int main(int argc, char **argv) {
    ecl_grid_type *grid = NULL;
    ensemble_config_type *ensemble_config = NULL;
    ecl_sum_type *refcase = NULL;

    enkf_obs_type *enkf_obs =
        enkf_obs_alloc(REFCASE_HISTORY, nullptr /* external_time_map */,
                       refcase, ensemble_config);

    obs_vector_type *obs_vector =
        obs_vector_alloc(SUMMARY_OBS, "WWCT", NULL, 2);
    summary_obs_type *summary_obs1 =
        summary_obs_alloc("SummaryKey", "ObservationKey", 43.2, 2.0);
    obs_vector_install_node(obs_vector, 0, summary_obs1);

    summary_obs_type *summary_obs2 =
        summary_obs_alloc("SummaryKey2", "ObservationKey2", 4.2, 0.1);
    obs_vector_install_node(obs_vector, 1, summary_obs2);

    obs_vector_type *obs_vector2 =
        obs_vector_alloc(SUMMARY_OBS, "WWCT2", NULL, 2);
    summary_obs_type *summary_obs3 =
        summary_obs_alloc("SummaryKey", "ObservationKey", 43.2, 2.0);
    obs_vector_install_node(obs_vector2, 0, summary_obs3);

    summary_obs_type *summary_obs4 =
        summary_obs_alloc("SummaryKey2", "ObservationKey2", 4.2, 0.1);
    obs_vector_install_node(obs_vector2, 1, summary_obs4);

    enkf_obs_add_obs_vector(enkf_obs, obs_vector);
    enkf_obs_add_obs_vector(enkf_obs, obs_vector2);

    enkf_obs_free(enkf_obs);

    exit(0);
}
