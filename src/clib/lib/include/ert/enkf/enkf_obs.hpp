#ifndef ERT_ENKF_OBS_H
#define ERT_ENKF_OBS_H
#include <time.h>

#include <ert/util/hash.h>
#include <ert/util/int_vector.h>
#include <ert/util/stringlist.h>

#include <ert/config/conf.hpp>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/obs_vector.hpp>
#include <ert/enkf/time_map.hpp>

extern "C" const char *enkf_obs_get_error(const enkf_obs_type *obs);

enkf_obs_type *enkf_obs_alloc(std::shared_ptr<TimeMap> external_time_map,
                              const ecl_sum_type *refcase);

extern "C" void enkf_obs_free(enkf_obs_type *enkf_obs);

extern "C" obs_vector_type *enkf_obs_iget_vector(const enkf_obs_type *obs,
                                                 int index);
extern "C" obs_vector_type *enkf_obs_get_vector(const enkf_obs_type *,
                                                const char *);
extern "C" void enkf_obs_add_obs_vector(enkf_obs_type *enkf_obs,
                                        const obs_vector_type *vector);

extern "C" obs_impl_type enkf_obs_get_type(const enkf_obs_type *enkf_obs,
                                           const char *key);

extern "C" stringlist_type *
enkf_obs_alloc_typed_keylist(enkf_obs_type *enkf_obs, obs_impl_type);
hash_type *enkf_obs_alloc_data_map(enkf_obs_type *enkf_obs);

extern "C" bool enkf_obs_has_key(const enkf_obs_type *, const char *);
extern "C" int enkf_obs_get_size(const enkf_obs_type *obs);

extern "C" stringlist_type *
enkf_obs_alloc_matching_keylist(const enkf_obs_type *enkf_obs,
                                const char *input_string);
extern "C" time_t enkf_obs_iget_obs_time(const enkf_obs_type *enkf_obs,
                                         int report_step);

#endif
