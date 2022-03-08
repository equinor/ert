#include <filesystem>
#include <iostream>
#include <fstream>
#include <optional>

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
#include <ert/enkf/row_scaling.hpp>

#include "../tmpdir.hpp"

void enkf_fs_umount(enkf_fs_type *fs);

namespace analysis {

std::optional<Eigen::MatrixXd>
load_parameters(enkf_fs_type *target_fs, ensemble_config_type *ensemble_config,
                const std::vector<int> &iens_active_index, int active_ens_size,
                const local_ministep_type *ministep);

void save_parameters(enkf_fs_type *target_fs,
                     ensemble_config_type *ensemble_config,
                     const std::vector<int> &iens_active_index,
                     const local_ministep_type *ministep,
                     const update_data_type &update_data);

std::vector<std::pair<Eigen::MatrixXd, std::shared_ptr<RowScaling>>>
load_row_scaling_parameters(enkf_fs_type *target_fs,
                            ensemble_config_type *ensemble_config,
                            const std::vector<int> &iens_active_index,
                            int active_ens_size,
                            const local_ministep_type *ministep);

} // namespace analysis

TEST_CASE("Write and read a matrix to enkf_fs instance",
          "[analysis][private]") {
    GIVEN("Saving a parameter matrix to enkf_fs instance") {
        WITH_TMPDIR;
        // create file system
        auto file_path = std::filesystem::current_path();
        auto fs = enkf_fs_create_fs(file_path.c_str(), BLOCK_FS_DRIVER_ID, NULL,
                                    true);

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

        // Creates files on fs where nodes are stored.
        // This is needed for the deserialization of the matrix, as the
        // enkf_fs instance has to instantiate the files were things are
        // stored.
        enkf_node_type *node = enkf_node_alloc(config_node);
        for (int i = 0; i < ensemble_size; i++) {
            enkf_node_store(node, fs, {.report_step = 0, .iens = i});
        }
        enkf_node_free(node);

        // set up ministep with one parmater and N realizations
        local_ministep_type *ministep =
            new local_ministep_type("not-important");
        ministep->add_active_data("TEST");
        std::vector<int> active_index;
        for (int i = 0; i < ensemble_size; i++) {
            active_index.push_back(i);
        }

        // Create matrix and save as as the parameter defined in the ministep
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(1, ensemble_size);
        for (int i = 0; i < ensemble_size; i++)
            A(0, i) = double(i) / 10.0;

        auto update_data = analysis::update_data_type(
            {}, {}, {}, {}, std::make_optional(A), {}, {});

        analysis::save_parameters(fs, ensemble_config, active_index, ministep,
                                  update_data);

        WHEN("loading parameters from enkf_fs") {
            auto B = analysis::load_parameters(
                fs, ensemble_config, active_index, ensemble_size, ministep);
            THEN("Loading parameters yield the same matrix") {
                REQUIRE(B.has_value());
                REQUIRE(A == B.value());
            }
        }

        //cleanup
        delete ministep;
        ensemble_config_free(ensemble_config);
        enkf_fs_decref(fs);
    }
}

TEST_CASE("Reading and writing matrices with rowscaling attached",
          "[analysis][private]") {
    GIVEN("Saving a parameter matrix to enkf_fs instance") {
        WITH_TMPDIR;
        // create file system
        auto file_path = std::filesystem::current_path();
        auto fs = enkf_fs_create_fs(file_path.c_str(), BLOCK_FS_DRIVER_ID, NULL,
                                    true);

        auto ensemble_config = ensemble_config_alloc_full("name-not-important");
        int ensemble_size = 10;
        // setting up a config node for a single parameter
        auto config_node =
            ensemble_config_add_gen_kw(ensemble_config, "TEST", false);
        // create template file
        std::ofstream templatefile("template");
        templatefile << "{\n\"a\": <COEFF_A>,\n\"b\": <COEFF_B>\n}"
                     << std::endl;
        templatefile.close();

        // create parameter_file
        std::ofstream paramfile("param");
        paramfile << "COEFF_A UNIFORM 0 1" << std::endl;
        paramfile << "COEFF_B UNIFORM 0 1" << std::endl;
        paramfile.close();

        enkf_config_node_update_gen_kw(config_node, "not_important.txt",
                                       "template", "param", nullptr, nullptr);

        // Creates files on fs where nodes are stored.
        // This is needed for the deserialization of the matrix, as the
        // enkf_fs instance has to instantiate the files were things are
        // stored.
        enkf_node_type *node = enkf_node_alloc(config_node);
        for (int i = 0; i < ensemble_size; i++) {
            enkf_node_store(node, fs, {.report_step = 0, .iens = i});
        }
        enkf_node_free(node);

        // set up ministep with one parmater and N realizations
        local_ministep_type *ministep =
            new local_ministep_type("not-important");
        ministep->add_active_data("TEST");

        // Must assing values to each row of the matrix
        auto scaling =
            local_ministep_get_or_create_row_scaling(ministep, "TEST");
        scaling->assign(0, 0.1);
        scaling->assign(1, 0.2);
        std::vector<int> active_index;
        for (int i = 0; i < ensemble_size; i++)
            active_index.push_back(i);

        // Create matrix and save as as the parameter defined in the ministep
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, ensemble_size);
        for (int i = 0; i < ensemble_size; i++) {
            A(0, i) = double(i) / 10.0;
            A(1, i) = double(i) / 20.0;
        }

        std::vector row_scaling_list{std::pair{A, scaling->shared_from_this()}};
        auto update_data = analysis::update_data_type({}, {}, {}, {}, {},
                                                      row_scaling_list, {});
        analysis::save_parameters(fs, ensemble_config, active_index, ministep,
                                  update_data);

        WHEN("loading parameters from enkf_fs") {
            auto parameter_matrices = analysis::load_row_scaling_parameters(
                fs, ensemble_config, active_index, ensemble_size, ministep);
            THEN("Loading parameters yield the same matrix") {
                for (int i = 0; i < parameter_matrices.size(); i++) {
                    auto A = parameter_matrices[i].first;
                    auto B = row_scaling_list[i].first;
                    REQUIRE(A == B);
                }
            }
        }

        //cleanup
        delete ministep;
        ensemble_config_free(ensemble_config);
        enkf_fs_decref(fs);
    }
}
