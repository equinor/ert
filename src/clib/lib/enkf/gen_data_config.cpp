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

#include <ert/enkf/config_keys.hpp>
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
    /** The key this gen_data instance is known under - needed for debugging. */
    char *key;
    /** The format used for loading gen_data instances when the forward model
     * has completed *AND* for loading the initial files.*/
    gen_data_file_format_type input_format;
    /** Data size, i.e. number of elements , indexed with report_step */
    int_vector_type *data_size_vector;
    /** The report steps where we expect to load data for this instance. */
    int_vector_type *active_report_steps;
};

gen_data_file_format_type
gen_data_config_get_input_format(const gen_data_config_type *config) {
    return config->input_format;
}

/*
   If current_size as queried from config->data_size_vector == -1
   (i.e. not set); we seek through
*/

int gen_data_config_get_data_size__(const gen_data_config_type *config,
                                    int report_step) {
    int current_size =
        int_vector_safe_iget(config->data_size_vector, report_step);
    return current_size;
}

static gen_data_config_type *gen_data_config_alloc(const char *key) {
    gen_data_config_type *config =
        (gen_data_config_type *)util_malloc(sizeof *config);

    config->key = util_alloc_string_copy(key);

    config->input_format = GEN_DATA_UNDEFINED;
    config->data_size_vector = int_vector_alloc(
        0, -1); /* The default value: -1 - indicates "NOT SET" */
    config->active_report_steps = int_vector_alloc(0, 0);

    return config;
}

gen_data_config_type *
gen_data_config_alloc_GEN_DATA_result(const char *key,
                                      gen_data_file_format_type input_format) {
    gen_data_config_type *config = gen_data_config_alloc(key);

    if (input_format == ASCII_TEMPLATE)
        util_abort("%s: Sorry can not use INPUT_FORMAT:ASCII_TEMPLATE\n",
                   __func__);

    if (input_format == GEN_DATA_UNDEFINED)
        util_abort("%s: Sorry must specify valid values for input format.\n",
                   __func__);

    config->input_format = input_format;
    return config;
}

void gen_data_config_free(gen_data_config_type *config) {
    int_vector_free(config->data_size_vector);
    int_vector_free(config->active_report_steps);

    free(config->key);
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

const char *gen_data_config_get_key(const gen_data_config_type *config) {
    return config->key;
}

VOID_FREE(gen_data_config)
