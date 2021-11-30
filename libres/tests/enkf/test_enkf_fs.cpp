#include <filesystem>

#include "catch2/catch.hpp"

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_obs.hpp>

#include "../tmpdir.hpp"

void enkf_fs_fwrite_misfit(enkf_fs_type *fs);
enkf_fs_type *enkf_fs_alloc_empty(const char *mount_point);
extern "C" void enkf_fs_umount(enkf_fs_type *fs);
extern "C" void
misfit_ensemble_initialize(misfit_ensemble_type *misfit_ensemble,
                           const ensemble_config_type *ensemble_config,
                           const enkf_obs_type *enkf_obs, enkf_fs_type *fs,
                           int ens_size, int history_length, bool force_init);
void enkf_fs_init_path_fmt(enkf_fs_type *fs);
void enkf_fs_set_read_only(enkf_fs_type *fs, bool read_only);

TEST_CASE("enkf_fs_fwrite_misfit", "[enkf]") {
    GIVEN("An instance of enkf_fs") {
        WITH_TMPDIR;
        auto file_path = std::filesystem::current_path();
        auto fs = enkf_fs_alloc_empty(file_path.c_str());
        enkf_fs_init_path_fmt(fs);

        WHEN("Misfits ensemble is initialized with minimal config") {
            auto misfit_ensemble = enkf_fs_get_misfit_ensemble(fs);
            auto ensemble_config =
                ensemble_config_alloc_full("name-not-important");
            auto enkf_obs =
                enkf_obs_alloc(nullptr, nullptr, nullptr, nullptr, nullptr);
            int ens_size = 1;
            int history_length = 1;
            misfit_ensemble_initialize(misfit_ensemble, ensemble_config,
                                       enkf_obs, fs, ens_size, history_length,
                                       false);

            THEN("Writing misfits creates a file") {
                enkf_fs_fwrite_misfit(fs);
                REQUIRE(std::filesystem::exists(file_path / "files" /
                                                "misfit-ensemble"));
            }
        }
        // Hack:
        // Want to call enkf_fs_umout to do cleanup.
        // However, by default it requires that fs->parameter, fs->dynamic_forecast and fs->index are set.
        // It is somewhat painful to set these parameters, so I instead set read_only to true
        // which bypasses accessing these members of fs.
        enkf_fs_set_read_only(fs, true);
        enkf_fs_umount(fs);
    }
}
