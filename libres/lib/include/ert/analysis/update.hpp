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

void enkf_main_save_parameters_from_ministep(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step, run_mode_type run_mode,
    enkf_state_type **ensemble, hash_type *use_count,
    const local_ministep_type *ministep,
    std::unordered_map<std::string, matrix_type *> parameters);

void enkf_main_save_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step,
    const local_ministep_type *ministep,
    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters);

std::unordered_map<std::string, matrix_type *>
enkf_main_load_parameters_from_ministep(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step, run_mode_type run_mode,
    meas_data_type *forecast, enkf_state_type **ensemble, hash_type *use_count,
    obs_data_type *obs_data, const local_ministep_type *ministep);

std::unordered_map<
    std::string,
    std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
enkf_main_load_row_scaling_parameters(
    enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
    int_vector_type *iens_active_index, int last_step, run_mode_type run_mode,
    meas_data_type *forecast, enkf_state_type **ensemble, hash_type *use_count,
    obs_data_type *obs_data, const local_ministep_type *ministep);

void enkf_main_analysis_update_with_rowscaling(
    analysis_module_type *module, const bool_vector_type *ens_mask,
    const meas_data_type *forecast, obs_data_type *obs_data,
    rng_type *shared_rng, matrix_type *E,
    std::unordered_map<
        std::string,
        std::vector<std::pair<matrix_type *, const row_scaling_type *>>>
        parameters);

void enkf_main_analysis_update_no_rowscaling(
    analysis_module_type *module, const bool_vector_type *ens_mask,
    const meas_data_type *forecast, obs_data_type *obs_data,
    rng_type *shared_rng, matrix_type *E,
    std::unordered_map<std::string, matrix_type *> parameters);

bool assert_update_viable(const analysis_config_type *analysis_config,
                          const enkf_fs_type *source_fs,
                          const int total_ens_size,
                          const local_updatestep_type *updatestep);

void copy_parameters(enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                     const ensemble_config_type *ensemble_config,
                     const int total_ens_size,
                     const int_vector_type *ens_active_list);
