#include <time.h>

#include <ert/util/vector.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_plot_genvector.hpp>
#include <ert/enkf/gen_data.hpp>

struct enkf_plot_genvector_struct {
    int iens;
    double_vector_type *data;
    const enkf_config_node_type *config_node;
};

enkf_plot_genvector_type *
enkf_plot_genvector_alloc(const enkf_config_node_type *config_node, int iens) {
    enkf_plot_genvector_type *vector =
        (enkf_plot_genvector_type *)util_malloc(sizeof *vector);
    vector->config_node = config_node;
    vector->data = double_vector_alloc(0, 0);
    vector->iens = iens;
    return vector;
}

void enkf_plot_genvector_free(enkf_plot_genvector_type *vector) {
    double_vector_free(vector->data);
    free(vector);
}

int enkf_plot_genvector_get_size(const enkf_plot_genvector_type *vector) {
    return double_vector_size(vector->data);
}

double enkf_plot_genvector_iget(const enkf_plot_genvector_type *vector,
                                int index) {
    return double_vector_iget(vector->data, index);
}

void enkf_plot_genvector_load(enkf_plot_genvector_type *vector,
                              enkf_fs_type *fs, int report_step) {
    enkf_node_type *work_node = enkf_node_alloc(vector->config_node);

    node_id_type node_id = {.report_step = report_step, .iens = vector->iens};

    if (enkf_node_try_load(work_node, fs, node_id)) {
        const gen_data_type *node =
            (const gen_data_type *)enkf_node_value_ptr(work_node);
        gen_data_copy_to_double_vector(node, vector->data);
    }
    enkf_node_free(work_node);
}
