/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'enkf_plot_gendata.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <float.h>

#include <ert/util/double_vector.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_plot_gendata.hpp>
#include <ert/enkf/obs_vector.hpp>

#define ENKF_PLOT_GENDATA_TYPE_ID 377626666

struct enkf_plot_gendata_struct {
    UTIL_TYPE_ID_DECLARATION;
    int size;
    int report_step;
    const enkf_config_node_type *enkf_config_node;
    enkf_plot_genvector_type **ensemble;
    double_vector_type *max_values;
    double_vector_type *min_values;
};

UTIL_IS_INSTANCE_FUNCTION(enkf_plot_gendata, ENKF_PLOT_GENDATA_TYPE_ID)

enkf_plot_gendata_type *
enkf_plot_gendata_alloc(const enkf_config_node_type *enkf_config_node) {
    if (enkf_config_node_get_impl_type(enkf_config_node) == GEN_DATA) {
        enkf_plot_gendata_type *data =
            (enkf_plot_gendata_type *)util_malloc(sizeof *data);
        UTIL_TYPE_ID_INIT(data, ENKF_PLOT_GENDATA_TYPE_ID);
        data->size = 0;
        data->enkf_config_node = enkf_config_node;
        data->ensemble = NULL;

        data->max_values = NULL;
        data->min_values = NULL;
        return data;
    } else {
        return NULL;
    }
}

enkf_plot_gendata_type *
enkf_plot_gendata_alloc_from_obs_vector(const obs_vector_type *obs_vector) {
    return enkf_plot_gendata_alloc(obs_vector_get_config_node(obs_vector));
}

void enkf_plot_gendata_free(enkf_plot_gendata_type *data) {
    for (int iens = 0; iens < data->size; iens++) {
        enkf_plot_genvector_free(data->ensemble[iens]);
    }

    free(data->ensemble);
    free(data);
}

int enkf_plot_gendata_get_size(const enkf_plot_gendata_type *data) {
    return data->size;
}

int enkf_plot_gendata_get_data_size(const enkf_plot_gendata_type *data) {
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(
            data->enkf_config_node);

    return gen_data_config_get_data_size__(gen_data_config, data->report_step);
}

std::vector<bool>
enkf_plot_gendata_active_mask(const enkf_plot_gendata_type *data) {
    gen_data_config_type *gen_data_config =
        (gen_data_config_type *)enkf_config_node_get_ref(
            data->enkf_config_node);
    const bool_vector_type *mask =
        gen_data_config_get_active_mask(gen_data_config);
    std::vector<bool> result;
    for (int i = 0; i < bool_vector_size(mask); i++) {
        result.push_back(bool_vector_iget(mask, i));
    }
    return result;
}

enkf_plot_genvector_type *
enkf_plot_gendata_iget(const enkf_plot_gendata_type *plot_data, int index) {
    if (index >= 0 && index < plot_data->size) {
        return plot_data->ensemble[index];
    } else {
        return NULL;
    }
}

static void enkf_plot_gendata_resize(enkf_plot_gendata_type *plot_gendata,
                                     int new_size) {
    if (new_size != plot_gendata->size) {
        int iens;

        if (new_size < plot_gendata->size) {
            for (iens = new_size; iens < plot_gendata->size; iens++) {
                enkf_plot_genvector_free(plot_gendata->ensemble[iens]);
            }
        }

        plot_gendata->ensemble = (enkf_plot_genvector_type **)util_realloc(
            plot_gendata->ensemble, new_size * sizeof *plot_gendata->ensemble);

        if (new_size > plot_gendata->size) {
            for (iens = plot_gendata->size; iens < new_size; iens++) {
                plot_gendata->ensemble[iens] = enkf_plot_genvector_alloc(
                    plot_gendata->enkf_config_node, iens);
            }
        }
        plot_gendata->size = new_size;
    }
}

void enkf_plot_gendata_load(enkf_plot_gendata_type *plot_data, enkf_fs_type *fs,
                            int report_step) {

    auto &state_map = enkf_fs_get_state_map(fs);
    int ens_size = state_map.size();

    const auto &mask = state_map.select_matching(STATE_HAS_DATA, true);
    enkf_plot_gendata_resize(plot_data, ens_size);
    plot_data->report_step = report_step;

    for (int iens = 0; iens < ens_size; iens++) {
        if (mask[iens]) {
            enkf_plot_genvector_type *vector =
                enkf_plot_gendata_iget(plot_data, iens);
            enkf_plot_genvector_load(vector, fs, report_step);
        }
    }
}

void enkf_plot_gendata_find_min_max_values__(
    enkf_plot_gendata_type *plot_data) {
    for (int iens = 0; iens < plot_data->size; iens++) {
        enkf_plot_genvector_type *vector =
            enkf_plot_gendata_iget(plot_data, iens);
        int size = enkf_plot_genvector_get_size(vector);
        if (iens == 0) {
            plot_data->min_values = double_vector_alloc(size, DBL_MAX);
            plot_data->max_values = double_vector_alloc(size, -DBL_MAX);
        }
        for (int index = 0; index < size; index++) {
            double value = enkf_plot_genvector_iget(vector, index);
            double_vector_iset(
                plot_data->min_values, index,
                util_double_min(
                    double_vector_iget(plot_data->min_values, index), value));
            double_vector_iset(
                plot_data->max_values, index,
                util_double_max(
                    double_vector_iget(plot_data->max_values, index), value));
        }
    }
}

double_vector_type *
enkf_plot_gendata_get_min_values(enkf_plot_gendata_type *plot_data) {
    if (plot_data->min_values == NULL) {
        enkf_plot_gendata_find_min_max_values__(plot_data);
    }
    return plot_data->min_values;
}

double_vector_type *
enkf_plot_gendata_get_max_values(enkf_plot_gendata_type *plot_data) {
    if (plot_data->max_values == NULL) {
        enkf_plot_gendata_find_min_max_values__(plot_data);
    }
    return plot_data->max_values;
}
