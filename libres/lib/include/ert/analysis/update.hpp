#include <stdexcept>
#include <ert/res_util/thread_pool.hpp>
#define HAVE_THREAD_POOL 1
#include <ert/res_util/matrix.hpp>
#include <ert/util/int_vector.h>
#include <ert/util/bool_vector.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/hash.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/meas_data.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/local_dataset.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/ensemble_config.hpp>

#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_obs.hpp>

/*
Store a set of parameters into a enkf_fs_type storage
*/
void analysis_save_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step,
    enkf_state_type **ensemble, hash_type *use_count,
    const local_ministep_type *ministep,
    std::unordered_map<std::string, matrix_type *> parameters);

/*
Store a set of row-scaled parameters into a enkf_fs_type storage
*/
void analysis_save_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step,
    const local_ministep_type *ministep,
    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters);

/*
load a set of parameters from a enkf_fs_type storage into a set of
matrices.
*/
std::unordered_map<std::string, matrix_type *> analysis_load_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step, meas_data_type *forecast,
    enkf_state_type **ensemble, hash_type *use_count, obs_data_type *obs_data,
    const local_ministep_type *ministep);

/*
load a set of parameters from a enkf_fs_type storage into a set of
matrices with the corresponding row-scaling object.
*/
std::unordered_map<
    std::string,
    std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
analysis_load_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step, meas_data_type *forecast,
    enkf_state_type **ensemble, hash_type *use_count, obs_data_type *obs_data,
    const local_ministep_type *ministep);

/*
Run the row-scaling enabled update algorithm on a set of A matrices.
*/
void analysis_run_analysis_update_with_rowscaling(
    analysis_module_type *module, const bool_vector_type *ens_mask,
    const meas_data_type *forecast, obs_data_type *obs_data,
    rng_type *shared_rng, matrix_type *E,
    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters);

/*
Run the update algorithm on a set of A matrices without row-scaling
*/
void analysis_run_analysis_update(
    analysis_module_type *module, const bool_vector_type *ens_mask,
    const meas_data_type *forecast, obs_data_type *obs_data,
    rng_type *shared_rng, matrix_type *E,
    std::unordered_map<std::string, matrix_type *> parameters);

/*
Check whether the current state and config allows the update algorithm
to be executed
*/
bool analysis_assert_update_viable(const analysis_config_type *analysis_config,
                                   const enkf_fs_type *source_fs,
                                   const int total_ens_size,
                                   const local_updatestep_type *updatestep);

/*
Copy all parameters from source_fs to target_fs
*/
void analysis_copy_parameters(enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                              const ensemble_config_type *ensemble_config,
                              const int total_ens_size,
                              const int_vector_type *ens_active_list);
