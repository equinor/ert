#ifndef __ENKF_OBS_H__
#define __ENKF_OBS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <sched_file.h>
#include <meas_matrix.h>
#include <well_obs.h>
#include <field_obs.h>
#include <ecl_rft_node.h>
#include <ensemble_config.h>
#include <enkf_config_node.h>
#include <gen_data_config.h>
#include <enkf_state.h>
#include <enkf_fs.h>
#include <enkf_types.h>

typedef struct enkf_obs_struct enkf_obs_type;


struct enkf_obs_struct {
  const history_type     * hist;
  const sched_file_type  * sched_file;
  hash_type              * obs_hash;
  int                      num_reports;
};


void          	         enkf_obs_free(enkf_obs_type * );
enkf_obs_type 	       * enkf_obs_fscanf_alloc(const char * , const ensemble_config_type * , const sched_file_type * ,const history_type * hist);
void                     enkf_obs_get_observations(enkf_obs_type * , int , obs_data_type * );
void          	         enkf_obs_measure_on_ensemble(const enkf_obs_type * , enkf_fs_type * , int , state_enum , int , const enkf_state_type ** , meas_matrix_type * );
void          	         enkf_obs_add_well_obs(enkf_obs_type *   , const enkf_config_node_type * , const char * , const char * , const char * );
bool                     enkf_obs_get_local_active(ensemble_config_type *, int );
gen_data_config_type   * enkf_obs_get_gen_data_config(ensemble_config_type *);
void                     enkf_obs_change_gen_data_config_iactive(ensemble_config_type *,int);
int                      enkf_obs_get_num_local_updates(ensemble_config_type *);
void                     enkf_obs_set_local_step(ensemble_config_type *, int);


#ifdef __cplusplus
}
#endif
#endif
