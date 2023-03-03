#include <cppitertools/enumerate.hpp>
#include <stdlib.h>

#include <ert/util/double_vector.h>
#include <ert/util/util.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/summary.hpp>

#define SUMMARY_UNDEF NAN

struct summary_struct {
    /** Can not be NULL - var_type is set on first load. */
    summary_config_type *config;
    double_vector_type *data_vector;
};

C_USED void summary_clear(summary_type *summary) {
    double_vector_reset(summary->data_vector);
}

summary_type *summary_alloc(const summary_config_type *summary_config) {
    summary_type *summary = (summary_type *)util_malloc(sizeof *summary);
    summary->config = (summary_config_type *)summary_config;
    summary->data_vector = double_vector_alloc(0, SUMMARY_UNDEF);
    return summary;
}

void summary_read_from_buffer(summary_type *summary, buffer_type *buffer,
                              enkf_fs_type *fs, int report_step) {
    enkf_util_assert_buffer_type(buffer, SUMMARY);
    {
        int size = buffer_fread_int(buffer);
        double default_value = buffer_fread_double(buffer);

        double_vector_set_default(summary->data_vector, default_value);
        double_vector_resize(summary->data_vector, size, default_value);
        buffer_fread(buffer, double_vector_get_ptr(summary->data_vector),
                     double_vector_element_size(summary->data_vector), size);
    }
}

bool summary_write_to_buffer(const summary_type *summary, buffer_type *buffer,
                             int report_step) {
    buffer_fwrite_int(buffer, SUMMARY);
    buffer_fwrite_int(buffer, double_vector_size(summary->data_vector));
    buffer_fwrite_double(buffer,
                         double_vector_get_default(summary->data_vector));
    buffer_fwrite(buffer, double_vector_get_ptr(summary->data_vector),
                  double_vector_element_size(summary->data_vector),
                  double_vector_size(summary->data_vector));
    return true;
}

C_USED bool summary_has_data(const summary_type *summary, int report_step) {
    return double_vector_size(summary->data_vector) > report_step;
}

void summary_free(summary_type *summary) {
    double_vector_free(summary->data_vector);
    free(summary);
}

void summary_serialize(const summary_type *summary, node_id_type node_id,
                       const ActiveList *active_list, Eigen::MatrixXd &A,
                       int row_offset, int column) {
    double value = summary_get(summary, node_id.report_step);
    enkf_matrix_serialize(&value, 1, ECL_DOUBLE, active_list, A, row_offset,
                          column);
}

void summary_deserialize(summary_type *summary, node_id_type node_id,
                         const ActiveList *active_list,
                         const Eigen::MatrixXd &A, int row_offset, int column) {
    double value;
    enkf_matrix_deserialize(&value, 1, ECL_DOUBLE, active_list, A, row_offset,
                            column);
    summary_set(summary, node_id.report_step, value);
}

int summary_length(const summary_type *summary) {
    return double_vector_size(summary->data_vector);
}

double summary_get(const summary_type *summary, int report_step) {
    return double_vector_iget(summary->data_vector, report_step);
}

void summary_set(summary_type *summary, int report_step, double value) {
    double_vector_iset(summary->data_vector, report_step, value);
}

double summary_undefined_value() { return SUMMARY_UNDEF; }

std::vector<double> summary_user_get_vector(const summary_type *summary) {
    std::vector<double> values(double_vector_size(summary->data_vector));
    for (int step = 0; step < double_vector_size(summary->data_vector); step++)
        values[step] = double_vector_iget(summary->data_vector, step);
    return values;
}

bool summary_forward_load_vector(summary_type *summary,
                                 const ecl_sum_type *ecl_sum,
                                 const std::vector<int> &time_index) {
    bool loadOK = false;

    if (ecl_sum == NULL)
        return false;

    const char *var_key = summary_config_get_var(summary->config);
    load_fail_type load_fail_action =
        summary_config_get_load_fail_mode(summary->config);
    bool normal_load = false;

    if (load_fail_action != LOAD_FAIL_EXIT) {
        // The load will always ~succeed - but if we do not have the data; we
        // will fill the vector with zeros.

        if (!ecl_sum_has_general_var(ecl_sum, var_key)) {
            for (auto summary_step : time_index)
                double_vector_iset(summary->data_vector, summary_step, 0);
            loadOK = true;

            if (load_fail_action == LOAD_FAIL_WARN)
                fprintf(
                    stderr,
                    "** WARNING ** Failed summary:%s does not have key:%s \n",
                    ecl_sum_get_case(ecl_sum), var_key);
        } else
            normal_load = true;
    }

    if (!normal_load)
        return loadOK;

    int key_index = ecl_sum_get_general_var_params_index(ecl_sum, var_key);
    for (auto [store_index, summary_index] : iter::enumerate(time_index)) {
        if (ecl_sum_has_report_step(ecl_sum, summary_index)) {
            int last_update_step_index =
                ecl_sum_iget_report_end(ecl_sum, summary_index);
            double_vector_iset(
                summary->data_vector, store_index,
                ecl_sum_iget(ecl_sum, last_update_step_index, key_index));
        }
    }
    return true;
}

VOID_ALLOC(summary)
VOID_FREE(summary)
VOID_WRITE_TO_BUFFER(summary)
VOID_READ_FROM_BUFFER(summary)
VOID_SERIALIZE(summary)
VOID_DESERIALIZE(summary)
VOID_CLEAR(summary)
VOID_HAS_DATA(summary)
