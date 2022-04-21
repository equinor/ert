/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'obs_data.c' is part of ERT - Ensemble based Reservoir Tool.

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

/*
See the file README.obs for ducumentation of the varios datatypes
involved with observations/measurement/+++.


The file contains two different variables holding the number of
observations, nrobs_total and nrobs_active. The first holds the total
number of observations at this timestep, and the second holds the
number of active measurements at this timestep; the inactive
measurements have been deactivated the obs_data_deactivate_outliers()
function.

The flow is as follows:

 1. All the observations have been collected in an obs_data instance,
    and all the corresponding measurements of the state have been
    collected in a meas_data instance - we are ready for analysis.

 2. The functions meas_data_alloc_stats() is called to calculate
    the ensemble mean and std of all the measurements.

 3. The function obs_data_deactivate_outliers() is called to compare
    the ensemble mean and std with the observations, in the case of
    outliers the number obs_active flag of the obs_data instance is
    set to false.

 4. The remaining functions (and matrices) now refer to the number of
    active observations, however the "raw" observations found in the
    obs_data instance are in a vector with nrobs_total observations;
    i.e. we must handle two indices and two total lengths. A bit
    messy.


Variables of size nrobs_total:
------------------------------
 o obs->value / obs->std / obs->obs_active
 o meanS , innov, stdS


variables of size nrobs_active:
-------------------------------
Matrices: S, D, E and various internal variables.
*/
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <vector>

#include <ert/util/util.h>
#include <ert/util/vector.h>

#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/obs_data.hpp>

#define OBS_BLOCK_TYPE_ID 995833

struct obs_block_struct {
    UTIL_TYPE_ID_DECLARATION;
    char *obs_key;
    int size;
    double *value;
    double *std;

    active_type *active_mode;
    int active_size;
    double global_std_scaling;
};

struct obs_data_struct {
    vector_type *data; /* vector with obs_block instances. */
    bool_vector_type *mask;
    double global_std_scaling;
};

static UTIL_SAFE_CAST_FUNCTION(obs_block, OBS_BLOCK_TYPE_ID)

    obs_block_type *obs_block_alloc(const char *obs_key, int obs_size,
                                    double global_std_scaling) {
    obs_block_type *obs_block =
        (obs_block_type *)util_malloc(sizeof *obs_block);

    UTIL_TYPE_ID_INIT(obs_block, OBS_BLOCK_TYPE_ID);
    obs_block->size = obs_size;
    obs_block->obs_key = util_alloc_string_copy(obs_key);
    obs_block->value =
        (double *)util_calloc(obs_size, sizeof *obs_block->value);
    obs_block->std = (double *)util_calloc(obs_size, sizeof *obs_block->std);
    obs_block->active_mode =
        (active_type *)util_calloc(obs_size, sizeof *obs_block->active_mode);
    obs_block->global_std_scaling = global_std_scaling;
    {
        for (int iobs = 0; iobs < obs_size; iobs++)
            obs_block->active_mode[iobs] = LOCAL_INACTIVE;
    }
    obs_block->active_size = 0;
    return obs_block;
}

void obs_block_free(obs_block_type *obs_block) {
    free(obs_block->obs_key);
    free(obs_block->value);
    free(obs_block->std);
    free(obs_block->active_mode);
    free(obs_block);
}

static void obs_block_free__(void *arg) {
    obs_block_type *obs_block = obs_block_safe_cast(arg);
    obs_block_free(obs_block);
}

void obs_block_deactivate(obs_block_type *obs_block, int iobs, bool verbose,
                          const char *msg) {
    if (obs_block->active_mode[iobs] == ACTIVE) {
        if (verbose)
            printf("Deactivating: %s(%d) : %s \n", obs_block->obs_key, iobs,
                   msg);
        obs_block->active_mode[iobs] = DEACTIVATED;
        obs_block->active_size--;
    }
}

const char *obs_block_get_key(const obs_block_type *obs_block) {
    return obs_block->obs_key;
}

void obs_block_iset(obs_block_type *obs_block, int iobs, double value,
                    double std) {
    obs_block->value[iobs] = value;
    obs_block->std[iobs] = std;
    if (obs_block->active_mode[iobs] != ACTIVE) {
        obs_block->active_mode[iobs] = ACTIVE;
        obs_block->active_size++;
    }
}

void obs_block_iset_missing(obs_block_type *obs_block, int iobs) {
    if (obs_block->active_mode[iobs] == ACTIVE)
        obs_block->active_size--;
    obs_block->active_mode[iobs] = MISSING;
}

double obs_block_iget_std(const obs_block_type *obs_block, int iobs) {
    return obs_block->std[iobs] * obs_block->global_std_scaling;
}

double obs_block_iget_value(const obs_block_type *obs_block, int iobs) {
    return obs_block->value[iobs];
}

active_type obs_block_iget_active_mode(const obs_block_type *obs_block,
                                       int iobs) {
    return obs_block->active_mode[iobs];
}

bool obs_block_iget_is_active(const obs_block_type *obs_block, int iobs) {
    return obs_block->active_mode[iobs] == ACTIVE;
}

int obs_block_get_size(const obs_block_type *obs_block) {
    return obs_block->size;
}

int obs_block_get_active_size(const obs_block_type *obs_block) {
    return obs_block->active_size;
}

static void obs_block_set_active_mask(const obs_block_type *obs_block,
                                      bool_vector_type *mask, int *offset) {
    for (int i = 0; i < obs_block->size; i++) {
        if (obs_block->active_mode[i] == ACTIVE)
            bool_vector_iset(mask, *offset, true);
        else
            bool_vector_iset(mask, *offset, false);
        (*offset)++;
    }
}

obs_data_type *obs_data_alloc(double global_std_scaling) {
    obs_data_type *obs_data = (obs_data_type *)util_malloc(sizeof *obs_data);
    obs_data->data = vector_alloc_new();
    obs_data->mask = bool_vector_alloc(0, false);
    obs_data->global_std_scaling = global_std_scaling;
    return obs_data;
}

obs_block_type *obs_data_add_block(obs_data_type *obs_data, const char *obs_key,
                                   int obs_size) {
    obs_block_type *new_block =
        obs_block_alloc(obs_key, obs_size, obs_data->global_std_scaling);
    vector_append_owned_ref(obs_data->data, new_block, obs_block_free__);
    return new_block;
}

obs_block_type *obs_data_iget_block(obs_data_type *obs_data, int index) {
    return (obs_block_type *)vector_iget(obs_data->data,
                                         index); // CXX_CAST_ERROR
}

const obs_block_type *obs_data_iget_block_const(const obs_data_type *obs_data,
                                                int index) {
    return (const obs_block_type *)vector_iget_const(obs_data->data,
                                                     index); // CXX_CAST_ERROR
}

void obs_data_free(obs_data_type *obs_data) {
    vector_free(obs_data->data);
    bool_vector_free(obs_data->mask);
    free(obs_data);
}

Eigen::VectorXd obs_data_values_as_vector(const obs_data_type *obs_data) {
    int active_obs_size = obs_data_get_active_size(obs_data);
    Eigen::VectorXd obs_values = Eigen::VectorXd::Zero(active_obs_size);
    int obs_offset = 0;
    for (int block_nr = 0; block_nr < vector_get_size(obs_data->data);
         block_nr++) {
        const obs_block_type *obs_block =
            (const obs_block_type *)vector_iget_const(obs_data->data, block_nr);
        for (int iobs = 0; iobs < obs_block->size; iobs++) {
            if (obs_block->active_mode[iobs] == ACTIVE) {
                obs_values(obs_offset) = obs_block->value[iobs];
                obs_offset++;
            }
        }
    }
    return obs_values;
}

Eigen::VectorXd obs_data_errors_as_vector(const obs_data_type *obs_data) {
    int active_obs_size = obs_data_get_active_size(obs_data);
    Eigen::VectorXd obs_errors = Eigen::VectorXd::Zero(active_obs_size);
    int obs_offset = 0;
    for (int block_nr = 0; block_nr < vector_get_size(obs_data->data);
         block_nr++) {
        const obs_block_type *obs_block =
            (const obs_block_type *)vector_iget_const(obs_data->data, block_nr);
        for (int iobs = 0; iobs < obs_block->size; iobs++) {
            if (obs_block->active_mode[iobs] == ACTIVE) {
                obs_errors(obs_offset) = obs_block->std[iobs];
                obs_offset++;
            }
        }
    }
    return obs_errors;
}

int obs_data_get_active_size(const obs_data_type *obs_data) {
    int active_size = 0;
    for (int block_nr = 0; block_nr < vector_get_size(obs_data->data);
         block_nr++) {
        const obs_block_type *obs_block =
            (const obs_block_type *)vector_iget_const(obs_data->data, block_nr);
        active_size += obs_block->active_size;
    }

    return active_size;
}

int obs_data_get_num_blocks(const obs_data_type *obs_data) {
    return vector_get_size(obs_data->data);
}

int obs_data_get_total_size(const obs_data_type *obs_data) {
    int total_size = 0;
    for (int block_nr = 0; block_nr < vector_get_size(obs_data->data);
         block_nr++) {
        const obs_block_type *obs_block =
            (const obs_block_type *)vector_iget_const(obs_data->data, block_nr);
        total_size += obs_block->size;
    }
    return total_size;
}

static const obs_block_type *
obs_data_lookup_block(const obs_data_type *obs_data, int total_index,
                      int *block_offset) {
    if (total_index < obs_data_get_total_size(obs_data)) {
        const obs_block_type *obs_block;
        int total_offset = 0;
        int block_index = 0;
        int block_size;

        while (true) {
            obs_block = (const obs_block_type *)vector_iget_const(
                obs_data->data, block_index);
            block_size = obs_block->size;
            if ((block_size + total_offset) > total_index)
                break;

            total_offset += block_size;
            block_index++;
        }
        *block_offset = total_offset;
        return obs_block;
    } else {
        util_abort("%s: could not lookup obs-block \n", __func__);
        return NULL;
    }
}

double obs_data_iget_value(const obs_data_type *obs_data, int total_index) {
    int total_offset;
    const obs_block_type *obs_block =
        obs_data_lookup_block(obs_data, total_index, &total_offset);
    return obs_block_iget_value(obs_block, total_index - total_offset);
}

double obs_data_iget_std(const obs_data_type *obs_data, int total_index) {
    int total_offset;
    const obs_block_type *obs_block =
        obs_data_lookup_block(obs_data, total_index, &total_offset);
    return obs_block_iget_std(obs_block, total_index - total_offset);
}

std::vector<bool> obs_data_get_active_mask(const obs_data_type *obs_data) {
    int total_size = obs_data_get_total_size(obs_data);
    bool_vector_resize(obs_data->mask, total_size,
                       false); //too account for extra data blocks added/removed

    int offset = 0;
    for (int block_nr = 0; block_nr < vector_get_size(obs_data->data);
         block_nr++) {
        const obs_block_type *obs_block =
            (const obs_block_type *)vector_iget_const(obs_data->data, block_nr);
        obs_block_set_active_mask(obs_block, obs_data->mask, &offset);
    }
    std::vector<bool> stl_mask{};
    for (int i = 0; i < bool_vector_size(obs_data->mask); i++) {
        stl_mask.push_back(bool_vector_iget(obs_data->mask, i));
    }
    return stl_mask;
}
