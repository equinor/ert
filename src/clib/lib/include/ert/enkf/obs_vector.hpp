#ifndef ERT_OBS_VECTOR_H
#define ERT_OBS_VECTOR_H

#include <time.h>
#include <vector>

#include <ert/util/bool_vector.h>
#include <ert/util/int_vector.h>

#include <ert/config/conf.hpp>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/time_map.hpp>

enum history_source_type {
    REFCASE_SIMULATED = 1, /** ecl_sum_get_well_var( "WWCT" );  */
    REFCASE_HISTORY = 2,   /** ecl_sum_get_well_var( "WWCTH" ); */
};

typedef enum { GEN_OBS = 1, SUMMARY_OBS = 2 } obs_impl_type;

typedef struct obs_vector_struct obs_vector_type;

typedef void(obs_free_ftype)(void *);
typedef void(obs_user_get_ftype)(void *, const char *, double *, double *,
                                 bool *);
typedef void(obs_update_std_scale_ftype)(void *, double, const ActiveList *);
typedef double(obs_chi2_ftype)(const void *, const void *, node_id_type);

extern "C" void obs_vector_free(obs_vector_type *);
extern "C" int obs_vector_get_num_active(const obs_vector_type *);
extern "C" bool obs_vector_iget_active(const obs_vector_type *, int);
extern "C" const char *obs_vector_get_state_kw(const obs_vector_type *);
extern "C" const char *obs_vector_get_key(const obs_vector_type *);
extern "C" obs_impl_type obs_vector_get_impl_type(const obs_vector_type *);
const std::vector<int> &obs_vector_get_step_list(const obs_vector_type *vector);
void obs_vector_user_get(const obs_vector_type *obs_vector,
                         const char *index_key, int report_step, double *value,
                         double *std, bool *valid);
extern "C" int obs_vector_get_next_active_step(const obs_vector_type *, int);
extern "C" void *obs_vector_iget_node(const obs_vector_type *, int);
extern "C" obs_vector_type *obs_vector_alloc(obs_impl_type obs_type,
                                             const char *obs_key,
                                             const char *config_node_key,
                                             size_t num_reports);

extern "C" void obs_vector_install_node(obs_vector_type *obs_vector,
                                        int obs_index, void *node);

extern "C" const char *
obs_vector_get_obs_key(const obs_vector_type *obs_vector);

VOID_FREE_HEADER(obs_vector);

#endif
