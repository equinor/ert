/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_obs.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ENKF_OBS_H
#define ERT_ENKF_OBS_H
#include <time.h>

#include <ert/util/hash.h>
#include <ert/util/int_vector.h>
#include <ert/util/stringlist.h>
#include <ert/util/type_macros.h>

#include <ert/config/conf.hpp>

#include <ert/sched/history.hpp>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/local_obsdata.hpp>
#include <ert/enkf/local_obsdata_node.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/obs_vector.hpp>
#include <ert/enkf/time_map.hpp>

bool enkf_obs_have_obs(const enkf_obs_type *enkf_obs);
extern "C" bool enkf_obs_is_valid(const enkf_obs_type *);

enkf_obs_type *enkf_obs_alloc(const history_type *history,
                              time_map_type *external_time_map,
                              const ecl_grid_type *grid,
                              const ecl_sum_type *refcase,
                              ensemble_config_type *ensemble_config);

extern "C" void enkf_obs_free(enkf_obs_type *enkf_obs);

extern "C" obs_vector_type *enkf_obs_iget_vector(const enkf_obs_type *obs,
                                                 int index);
extern "C" obs_vector_type *enkf_obs_get_vector(const enkf_obs_type *,
                                                const char *);
extern "C" void enkf_obs_add_obs_vector(enkf_obs_type *enkf_obs,
                                        const obs_vector_type *vector);

void enkf_obs_load(enkf_obs_type *, const char *, double);
extern "C" void enkf_obs_clear(enkf_obs_type *enkf_obs);
extern "C" obs_impl_type enkf_obs_get_type(const enkf_obs_type *enkf_obs,
                                           const char *key);

void enkf_obs_get_obs_and_measure_data(const enkf_obs_type *enkf_obs,
                                       enkf_fs_type *fs,
                                       const LocalObsData *local_obsdata,
                                       const std::vector<int> &ens_active_list,
                                       meas_data_type *meas_data,
                                       obs_data_type *obs_data);

extern "C" stringlist_type *
enkf_obs_alloc_typed_keylist(enkf_obs_type *enkf_obs, obs_impl_type);
hash_type *enkf_obs_alloc_data_map(enkf_obs_type *enkf_obs);

extern "C" bool enkf_obs_has_key(const enkf_obs_type *, const char *);
extern "C" int enkf_obs_get_size(const enkf_obs_type *obs);

hash_iter_type *enkf_obs_alloc_iter(const enkf_obs_type *enkf_obs);

extern "C" stringlist_type *
enkf_obs_alloc_matching_keylist(const enkf_obs_type *enkf_obs,
                                const char *input_string);
extern "C" time_t enkf_obs_iget_obs_time(const enkf_obs_type *enkf_obs,
                                         int report_step);
void enkf_obs_add_local_nodes_with_data(const enkf_obs_type *enkf_obs,
                                        LocalObsData *local_obs,
                                        enkf_fs_type *fs,
                                        const bool_vector_type *ens_mask);
conf_class_type *enkf_obs_get_obs_conf_class();
UTIL_IS_INSTANCE_HEADER(enkf_obs);

#endif
