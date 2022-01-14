#include <filesystem>
#include <iostream>
#include <fstream>

#include "catch2/catch.hpp"

#include <ert/enkf/enkf_fs.hpp>
#include <ert/analysis/update.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/local_ministep.hpp>
#include <ert/util/type_vector_functions.hpp>

#include "../tmpdir.hpp"

namespace analysis {
void copy_parameters(enkf_fs_type *source_fs, enkf_fs_type *target_fs,
                     const ensemble_config_type *ensemble_config,
                     const int_vector_type *ens_active_list);
} // namespace analysis

TEST_CASE("Copy parameters from one source-fs to target-fs",
          "[analysis][private]") {
    GIVEN("A parameter stored on source-fs") {
        WITH_TMPDIR;
        // create file system
        auto source_path =
            std::filesystem::current_path() / std::filesystem::path("source");
        auto fs_source = enkf_fs_create_fs(source_path.c_str(),
                                           BLOCK_FS_DRIVER_ID, NULL, true);
        auto target_path =
            std::filesystem::current_path() / std::filesystem::path("target");
        auto fs_target = enkf_fs_create_fs(target_path.c_str(),
                                           BLOCK_FS_DRIVER_ID, NULL, true);

        auto ensemble_config = ensemble_config_alloc_full("name-not-important");
        int ensemble_size = 10;
        // setting up a config node for a single parameter
        auto config_node =
            ensemble_config_add_gen_kw(ensemble_config, "TEST", false);
        // create template file
        std::ofstream templatefile("template");
        templatefile << "{\n\"a\": <COEFF>\n}" << std::endl;
        templatefile.close();

        // create parameter_file
        std::ofstream paramfile("param");
        paramfile << "COEFF UNIFORM 0 1" << std::endl;
        paramfile.close();

        enkf_config_node_update_gen_kw(config_node, "not_important.txt",
                                       "template", "param", nullptr, nullptr);

        // Creates files on fs_source where nodes are stored.
        // This is needed for the deserialization of the matrix, as the
        // enkf_fs instance has to instantiate the files were things are
        // stored.
        enkf_node_type *node = enkf_node_alloc(config_node);
        for (int i = 0; i < ensemble_size; i++) {
            enkf_node_store(node, fs_source, {.report_step = 0, .iens = i});
        }
        enkf_node_free(node);

        int_vector_type *active_index = int_vector_alloc(ensemble_size, -1);
        for (int i = 0; i < ensemble_size; i++) {
            int_vector_iset(active_index, i, i);
        }
        WHEN("not copying parameters from source to target") {
            THEN("target fs has no data at the same locations") {
                enkf_node_type *node = enkf_node_alloc(config_node);
                for (int i = 0; i < ensemble_size; i++) {
                    REQUIRE(!enkf_node_has_data(node, fs_target,
                                                {.report_step = 0, .iens = i}));
                }
                enkf_node_free(node);
            }
        }
        WHEN("copying parameters from source to target") {
            analysis::copy_parameters(fs_source, fs_target, ensemble_config,
                                      active_index);
            THEN("target fs has data at the same locations") {
                enkf_node_type *node = enkf_node_alloc(config_node);
                for (int i = 0; i < ensemble_size; i++) {
                    REQUIRE(enkf_node_has_data(node, fs_target,
                                               {.report_step = 0, .iens = i}));
                }
                enkf_node_free(node);
            }
        }

        //cleanup
        int_vector_free(active_index);
        ensemble_config_free(ensemble_config);
        enkf_fs_decref(fs_source);
        enkf_fs_decref(fs_target);
    }
}
