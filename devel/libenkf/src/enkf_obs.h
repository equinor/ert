#ifndef __ENKF_OBS_H__
#define __ENKF_OBS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <history.h>
#include <enkf_fs.h>
#include <enkf_state.h>
#include <meas_matrix.h>
#include <obs_node.h>
#include <obs_data.h>

typedef struct enkf_obs_struct enkf_obs_type;



/** TODO
    Currently, this has to be here since enkf_state accesses obs_hash explicitly. 

    THIS SHOULD BE FIXED.
*/
struct enkf_obs_struct {
  /** A hash of obs_node_types indexed by user provided keys. */
  hash_type              * obs_hash; 
};








void enkf_obs_free(
  enkf_obs_type * enkf_obs);

void enkf_obs_add_obs(
  enkf_obs_type       * enkf_obs,
  const char          * key ,
  const obs_node_type * node);

void enkf_obs_get_observations(
  enkf_obs_type * enkf_obs ,
  int             report_step,
  obs_data_type * obs_data);

enkf_obs_type * enkf_obs_fscanf_alloc(
  const char         * config_file,
  const history_type * hist);

void enkf_obs_measure_on_ensemble(
        const enkf_obs_type    * enkf_obs,
        enkf_fs_type           * fs,
        int                      report_step,
        state_enum               state,
        int                      ens_size,
        const enkf_state_type ** ensemble ,
        meas_matrix_type       * meas_matrix);

#ifdef __cplusplus
}
#endif
#endif
