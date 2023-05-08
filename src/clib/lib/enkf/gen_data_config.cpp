#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/ecl/ecl_util.h>
#include <ert/logging.hpp>
#include <ert/util/bool_vector.h>
#include <ert/util/int_vector.h>
#include <ert/util/util.h>
#include <pybind11/stl.h>

#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/gen_data_config.hpp>

static auto logger = ert::get_logger("enkf");

/*
   About deactivating by the forward model
   ---------------------------------------

   For the gen_data instances the forward model has the capability to
   deactivate elements in a gen_data vector. This is implemented in
   the function gen_data_ecl_load which will look for a file with
   extension "_data" and then activate / deactivate elements
   accordingly.
*/

struct gen_data_config_struct {
    /** The report steps where we expect to load data for this instance. */
    int_vector_type *active_report_steps;
};

static gen_data_config_type *gen_data_config_alloc() {
    gen_data_config_type *config =
        (gen_data_config_type *)util_malloc(sizeof *config);

    config->active_report_steps = int_vector_alloc(0, 0);

    return config;
}

gen_data_config_type *gen_data_config_alloc_GEN_DATA_result() {
    gen_data_config_type *config = gen_data_config_alloc();
    return config;
}

void gen_data_config_free(gen_data_config_type *config) {
    int_vector_free(config->active_report_steps);
    free(config);
}

int gen_data_config_num_report_step(const gen_data_config_type *config) {
    return int_vector_size(config->active_report_steps);
}

bool gen_data_config_has_report_step(const gen_data_config_type *config,
                                     int report_step) {
    return int_vector_contains_sorted(config->active_report_steps, report_step);
}

void gen_data_config_add_report_step(gen_data_config_type *config,
                                     int report_step) {
    if (!gen_data_config_has_report_step(config, report_step)) {
        int_vector_append(config->active_report_steps, report_step);
        int_vector_sort(config->active_report_steps);
    }
}

int gen_data_config_iget_report_step(const gen_data_config_type *config,
                                     int index) {
    return int_vector_iget(config->active_report_steps, index);
}

VOID_FREE(gen_data_config)
