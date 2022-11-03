#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_plot_data.hpp>
#include <ert/enkf/enkf_plot_tvector.hpp>

struct enkf_plot_data_struct {
    const enkf_config_node_type *config_node;
    int size;
    enkf_plot_tvector_type **ensemble;
};

void enkf_plot_data_free(enkf_plot_data_type *plot_data) {
    int iens;
    for (iens = 0; iens < plot_data->size; iens++) {
        enkf_plot_tvector_free(plot_data->ensemble[iens]);
    }
    free(plot_data->ensemble);
    free(plot_data);
}

enkf_plot_data_type *
enkf_plot_data_alloc(const enkf_config_node_type *config_node, int size) {
    enkf_plot_data_type *plot_data =
        (enkf_plot_data_type *)util_malloc(sizeof *plot_data);
    plot_data->config_node = config_node;
    plot_data->size = size;
    plot_data->ensemble = (enkf_plot_tvector_type **)util_realloc(
        plot_data->ensemble, size * sizeof *plot_data->ensemble);
    for (int iens = 0; iens < size; iens++) {
        plot_data->ensemble[iens] =
            enkf_plot_tvector_alloc(plot_data->config_node, iens);
    }
    return plot_data;
}

enkf_plot_tvector_type *
enkf_plot_data_iget(const enkf_plot_data_type *plot_data, int index) {
    return plot_data->ensemble[index];
}
