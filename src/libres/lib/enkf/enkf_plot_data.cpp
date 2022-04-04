/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'enkf_plot_data.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_plot_data.hpp>
#include <ert/enkf/enkf_plot_tvector.hpp>

#define ENKF_PLOT_DATA_TYPE_ID 3331063

struct enkf_plot_data_struct {
    UTIL_TYPE_ID_DECLARATION;
    const enkf_config_node_type *config_node;
    int size;
    enkf_plot_tvector_type **ensemble;
};

static void enkf_plot_data_resize(enkf_plot_data_type *plot_data,
                                  int new_size) {
    if (new_size != plot_data->size) {
        int iens;

        if (new_size < plot_data->size) {
            for (iens = new_size; iens < plot_data->size; iens++) {
                enkf_plot_tvector_free(plot_data->ensemble[iens]);
            }
        }

        plot_data->ensemble = (enkf_plot_tvector_type **)util_realloc(
            plot_data->ensemble, new_size * sizeof *plot_data->ensemble);

        if (new_size > plot_data->size) {
            for (iens = plot_data->size; iens < new_size; iens++) {
                plot_data->ensemble[iens] =
                    enkf_plot_tvector_alloc(plot_data->config_node, iens);
            }
        }
        plot_data->size = new_size;
    }
}

static void enkf_plot_data_reset(enkf_plot_data_type *plot_data) {
    int iens;
    for (iens = 0; iens < plot_data->size; iens++) {
        enkf_plot_tvector_reset(plot_data->ensemble[iens]);
    }
}

void enkf_plot_data_free(enkf_plot_data_type *plot_data) {
    int iens;
    for (iens = 0; iens < plot_data->size; iens++) {
        enkf_plot_tvector_free(plot_data->ensemble[iens]);
    }
    free(plot_data->ensemble);
    free(plot_data);
}

UTIL_IS_INSTANCE_FUNCTION(enkf_plot_data, ENKF_PLOT_DATA_TYPE_ID);

enkf_plot_data_type *
enkf_plot_data_alloc(const enkf_config_node_type *config_node) {
    enkf_plot_data_type *plot_data =
        (enkf_plot_data_type *)util_malloc(sizeof *plot_data);
    UTIL_TYPE_ID_INIT(plot_data, ENKF_PLOT_DATA_TYPE_ID);
    plot_data->config_node = config_node;
    plot_data->size = 0;
    plot_data->ensemble = NULL;
    return plot_data;
}

enkf_plot_tvector_type *
enkf_plot_data_iget(const enkf_plot_data_type *plot_data, int index) {
    return plot_data->ensemble[index];
}

int enkf_plot_data_get_size(const enkf_plot_data_type *plot_data) {
    return plot_data->size;
}

void enkf_plot_data_load(enkf_plot_data_type *plot_data, enkf_fs_type *fs,
                         const char *index_key) {
    auto &state_map = enkf_fs_get_state_map(fs);
    int ens_size = state_map.size();

    std::vector<bool> mask = state_map.select_matching(STATE_HAS_DATA, true);
    enkf_plot_data_resize(plot_data, ens_size);
    enkf_plot_data_reset(plot_data);
    {
        for (int iens = 0; iens < ens_size; iens++) {
            if (mask[iens]) {
                enkf_plot_tvector_type *vector =
                    enkf_plot_data_iget(plot_data, iens);
                enkf_plot_tvector_load(vector, fs, index_key);
            }
        }
    }
}
