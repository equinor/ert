#include <stdexcept>
#define HAVE_THREAD_POOL 1
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/obs_data.hpp>
#include <ert/enkf/local_updatestep.hpp>
#include <ert/enkf/ensemble_config.hpp>

#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/enkf_obs.hpp>

namespace analysis {
bool smoother_update(std::vector<int> step_list,
                     const local_updatestep_type *updatestep,
                     int total_ens_size, enkf_obs_type *obs,
                     rng_type *shared_rng,
                     const analysis_config_type *analysis_config,
                     ensemble_config_type *ensemble_config,
                     enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                     FILE *log_stream, bool verbose);

} // namespace analysis
