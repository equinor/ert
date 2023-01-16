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
    pthread_mutex_t update_lock;
    /* All the fields below this line are related to the capability of the
     * forward model to deactivate elements in a gen_data instance. See
     * documentation above. */
    int ens_size;
    bool mask_modified;
    bool_vector_type *active_mask;
    int active_report_step;
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

int gen_data_config_get_data_size(const gen_data_config_type *config,
                                  int report_step) {
    int current_size = gen_data_config_get_data_size__(config, report_step);
    if (current_size < 0)
        throw pybind11::value_error("No data has been loaded for report step");
    return current_size;
}

int gen_data_config_get_initial_size(const gen_data_config_type *config) {
    int initial_size = int_vector_safe_iget(config->data_size_vector, 0);
    if (initial_size < 0)
        initial_size = 0;

    return initial_size;
}

static gen_data_config_type *gen_data_config_alloc(const char *key) {
    gen_data_config_type *config =
        (gen_data_config_type *)util_malloc(sizeof *config);

    config->key = util_alloc_string_copy(key);

    config->input_format = GEN_DATA_UNDEFINED;
    config->data_size_vector = int_vector_alloc(
        0, -1); /* The default value: -1 - indicates "NOT SET" */
    config->active_report_steps = int_vector_alloc(0, 0);
    config->active_mask = bool_vector_alloc(
        0,
        true); /* Elements are explicitly set to FALSE - this MUST default to true. */
    config->active_report_step = -1;
    config->ens_size = -1;
    pthread_mutex_init(&config->update_lock, NULL);

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

const bool_vector_type *
gen_data_config_get_active_mask(const gen_data_config_type *config) {
    return config->active_mask;
}

/**
 * @brief parses format string as a gen_data_file_format_type

   This function takes a string representation of one of the
   gen_data_file_format_type values, and returns the corresponding
   gen_data_file_format value. The recognized strings are

   * "ASCII"
   * "ASCII_TEMPLATE"

   Its the inverse action of gen_data_config_format_name

   @see gen_data_config_format_name
   @param format_string The file format string, ie. "ASCII"
   @return GEN_DATA_UNDEFINED if the string is not recognized or NULL, otherwise
      the corresponding gen_data_file_format_type, ie. ASCII.
*/
gen_data_file_format_type
gen_data_config_check_format(const char *format_string) {
    gen_data_file_format_type type = GEN_DATA_UNDEFINED;

    if (format_string != NULL) {

        if (strcmp(format_string, "ASCII") == 0)
            type = ASCII;
        else if (strcmp(format_string, "ASCII_TEMPLATE") == 0)
            type = ASCII_TEMPLATE;
    }

    return type;
}

/**
   The valid options are:

   INPUT_FORMAT:(ASCII|ASCII_TEMPLATE)
   RESULT_FILE:<filename to read EnKF <== Forward model>

*/
void gen_data_config_free(gen_data_config_type *config) {
    int_vector_free(config->data_size_vector);
    int_vector_free(config->active_report_steps);

    free(config->key);
    bool_vector_free(config->active_mask);

    free(config);
}

/**
   This function gets a size (from a gen_data) instance, and verifies
   that the size agrees with the currently stored size and
   report_step. If the report_step is new we just record the new info,
   otherwise it will break hard.

   Does not work properly with:

   1. keep_run_path - the load_file will be left hanging around - and loaded again and again.
   2. Doing forward several steps - how to (time)index the files?

*/
void gen_data_config_assert_size(gen_data_config_type *config, int data_size,
                                 int report_step) {
    pthread_mutex_lock(&config->update_lock);
    {
        int current_size =
            int_vector_safe_iget(config->data_size_vector, report_step);
        if (current_size < 0) {
            int_vector_iset(config->data_size_vector, report_step, data_size);
            current_size = data_size;
        }

        if (current_size != data_size) {
            util_abort("%s: Size mismatch when loading:%s from file - got %d "
                       "elements - expected:%d [report_step:%d] \n",
                       __func__, gen_data_config_get_key(config), data_size,
                       current_size, report_step);
        }
    }
    pthread_mutex_unlock(&config->update_lock);
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

const int_vector_type *
gen_data_config_get_active_report_steps(const gen_data_config_type *config) {
    return config->active_report_steps;
}

void gen_data_config_set_ens_size(gen_data_config_type *config, int ens_size) {
    config->ens_size = ens_size;
}

bool gen_data_config_valid_result_format(const char *result_file_fmt) {
    if (result_file_fmt) {
        if (util_is_abs_path(result_file_fmt))
            return false;
        else {
            if (util_int_format_count(result_file_fmt) == 1)
                return true;
            else
                return false;
        }
    } else
        return false;
}

const char *gen_data_config_get_key(const gen_data_config_type *config) {
    return config->key;
}

/**
 * @brief returns the format string correspondng to the gen_data_file_format_type.

   This function takes a gen_data_file_format_type and returns its string representation.

   Its the inverse action of gen_data_config_check_format

   @see gen_data_config_check_format
   @param format_string The file format string, ie. "ASCII"
   @return GEN_DATA_UNDEFINED if the string is not recognized or NULL, otherwise
      the corresponding gen_data_file_format_type, ie. ASCII.
*/
static const char *
gen_data_config_format_name(gen_data_file_format_type format_type) {
    switch (format_type) {
    case GEN_DATA_UNDEFINED:
        return "UNDEFINED";
    case ASCII:
        return "ASCII";
    case ASCII_TEMPLATE:
        return "ASCII_TEMPLATE";
    default:
        util_abort("%s: What the f.. \n", __func__);
        return NULL;
    }
}

VOID_FREE(gen_data_config)
