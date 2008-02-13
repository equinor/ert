#ifndef __ENKF_OBS_H__
#define __ENKF_OBS_H__
#include <sched_file.h>
#include <meas_vector.h>
#include <well_obs.h>
#include <field_obs.h>
#include <ecl_rft_node.h>
#include <enkf_config.h>
#include <enkf_config_node.h>

typedef struct enkf_obs_struct enkf_obs_type;


struct enkf_obs_struct {
  const history_type     * hist;
  const sched_file_type  * sched_file;
  hash_type              * obs_hash;
  int                      num_reports;
};


void            enkf_obs_free(enkf_obs_type * );
enkf_obs_type * enkf_obs_fscanf_alloc(const enkf_config_type * , const sched_file_type * ,const history_type * hist);

/*void 		enkf_obs_measure(enkf_obs_type * , int , const enkf_state_type *);*/
void 		enkf_obs_get_observations(enkf_obs_type * , int , obs_data_type * );
void            enkf_obs_add_well_obs(enkf_obs_type *   , const enkf_config_node_type * , const char * , const char * , const char * );
void            enkf_obs_add_field_obs(enkf_obs_type *  , const enkf_config_node_type * , const char * , const char * , int , const int * , const int *, const int *, const double * , time_t );
void            enkf_obs_add_rft_obs(enkf_obs_type *    , const enkf_config_node_type * , const ecl_rft_node_type * , const double * );
/*enkf_obs_type * enkf_obs_fscanf_alloc(const char *    , const enkf_ens_type * , const history_type * );*/

#endif
