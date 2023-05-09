#ifndef ERT_GEN_OBS_H
#define ERT_GEN_OBS_H

#include <ert/enkf/active_list.hpp>
#include <ert/enkf/enkf_macros.hpp>

typedef struct gen_obs_struct gen_obs_type;

gen_obs_type *gen_obs_alloc(const char *obs_key, const char *, double, double,
                            const char *, const char *);
extern "C" gen_obs_type *
gen_obs_alloc__(const char *obs_key); // for python bindings
void gen_obs_user_get_with_data_index(const gen_obs_type *gen_obs,
                                      const char *index_key, double *value,
                                      double *std, bool *valid);

void gen_obs_update_std_scale(gen_obs_type *gen_obs, double std_multiplier,
                              const ActiveList *active_list);
extern "C" int gen_obs_get_size(const gen_obs_type *gen_obs);
extern "C" double gen_obs_iget_std(const gen_obs_type *gen_obs, int index);
extern "C" void gen_obs_load_std(const gen_obs_type *gen_obs, int size,
                                 double *data);
extern "C" double gen_obs_iget_value(const gen_obs_type *gen_obs, int index);
extern "C" void gen_obs_load_values(const gen_obs_type *gen_obs, int size,
                                    double *data);
extern "C" double gen_obs_iget_std_scaling(const gen_obs_type *gen_obs,
                                           int index);
extern "C" PY_USED int gen_obs_get_obs_index(const gen_obs_type *gen_obs,
                                             int index);
extern "C" void gen_obs_load_observation(gen_obs_type *gen_obs,
                                         const char *obs_file);
extern "C" void gen_obs_set_scalar(gen_obs_type *gen_obs, double scalar_value,
                                   double scalar_std);
extern "C" void gen_obs_attach_data_index(gen_obs_type *gen_obs,
                                          const int_vector_type *data_index);
extern "C" void gen_obs_load_data_index(gen_obs_type *obs,
                                        const char *data_index_file);
void gen_obs_parse_data_index(gen_obs_type *obs, const char *data_index_string);
extern "C" void gen_obs_free(gen_obs_type *gen_obs);

VOID_FREE_HEADER(gen_obs);
VOID_USER_GET_OBS_HEADER(gen_obs);
VOID_UPDATE_STD_SCALE_HEADER(gen_obs);

#endif
