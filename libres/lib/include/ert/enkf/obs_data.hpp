/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'obs_data.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_OBS_DATA_H
#define ERT_OBS_DATA_H

#include <stdbool.h>
#include <stdio.h>
#include <vector>

#include <ert/util/hash.h>
#include <ert/util/rng.h>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/meas_data.hpp>

typedef struct obs_data_struct obs_data_type;
typedef struct obs_block_struct obs_block_type;
extern "C" void obs_block_free(obs_block_type *obs_block);
active_type obs_block_iget_active_mode(const obs_block_type *obs_block,
                                       int iobs);
extern "C" obs_block_type *obs_block_alloc(const char *obs_key, int obs_size,
                                           double global_std_scaling);
extern "C" int obs_block_get_active_size(const obs_block_type *obs_block);

void obs_block_deactivate(obs_block_type *obs_block, int iobs, bool verbose,
                          const char *msg);
extern "C" int obs_block_get_size(const obs_block_type *obs_block);
extern "C" void obs_block_iset(obs_block_type *obs_block, int iobs,
                               double value, double std);
void obs_block_iset_missing(obs_block_type *obs_block, int iobs);

extern "C" double obs_block_iget_std(const obs_block_type *obs_block, int iobs);
extern "C" double obs_block_iget_value(const obs_block_type *obs_block,
                                       int iobs);

extern "C" obs_block_type *obs_data_iget_block(obs_data_type *obs_data,
                                               int index);
const obs_block_type *obs_data_iget_block_const(const obs_data_type *obs_data,
                                                int block_nr);
extern "C" obs_block_type *
obs_data_add_block(obs_data_type *obs_data, const char *obs_key, int obs_size);

extern "C" obs_data_type *obs_data_alloc(double global_std_scaling);
extern "C" void obs_data_free(obs_data_type *);

Eigen::VectorXd obs_data_values_as_vector(const obs_data_type *obs_data);
Eigen::VectorXd obs_data_errors_as_vector(const obs_data_type *obs_data);

extern "C" int obs_data_get_active_size(const obs_data_type *obs_data);
extern "C" int obs_data_get_total_size(const obs_data_type *obs_data);
extern "C" int obs_data_get_num_blocks(const obs_data_type *obs_data);
extern "C" const char *obs_block_get_key(const obs_block_type *obs_block);
extern "C" double obs_data_iget_value(const obs_data_type *obs_data,
                                      int total_index);
extern "C" PY_USED double obs_data_iget_std(const obs_data_type *obs_data,
                                            int total_index);
extern "C" PY_USED bool
obs_block_iget_is_active(const obs_block_type *obs_block, int iobs);

std::vector<bool> obs_data_get_active_mask(const obs_data_type *obs_data);

#endif
