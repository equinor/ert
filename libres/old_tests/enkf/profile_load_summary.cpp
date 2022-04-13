#include <stdlib.h>
#include <filesystem>
#include <chrono>

#include <ert/util/type_vector_functions.hpp>
#include <ert/enkf/res_config.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_main.hpp>
#include <ert/enkf/enkf_plot_data.hpp>

extern "C" void set_site_config(const char *site_config);

int main(int argc, char **argv) {
    std::filesystem::current_path("/Users/bjarne/git/Equinor/ert/Testing/2650/"
                                  "drogon_design_560_realizations/ert/model");
    printf("\nCurrent path: %s\n", std::filesystem::current_path().c_str());

    //	const char *config_file = "/Users/bjarne/git/Equinor/ert/Testing/2650/drogon_design_560_realizations/ert/model/drogon_design_bgh.ert";
    const char *config_file = "drogon_design_bgh.ert";
    set_site_config(config_file);
    //    char *model_config;
    //    util_alloc_file_components(config_file, NULL, &model_config, NULL);
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        res_config_type *res_config = res_config_alloc_load(config_file);
        enkf_main_type *enkf_main = enkf_main_alloc(res_config, false, false);

        ensemble_config_type *ensemble_config =
            enkf_main_get_ensemble_config(enkf_main);
        enkf_config_node_type *ensemble_config_node =
            ensemble_config_get_node(ensemble_config, "FGOR");

        enkf_node_type *node_type = enkf_node_alloc(ensemble_config_node);

        ert_impl_type impltype = enkf_node_get_impl_type(node_type);

        enkf_plot_data_type *plot_data =
            enkf_plot_data_alloc(ensemble_config_node);

        enkf_fs_type *default_fs =
            enkf_main_mount_alt_fs(enkf_main, "default", false);

        //        summary_key_matcher_type *key_matcher =
        //        		ensemble_config_get_summary_key_matcher(ensemble_config);

        stringlist_type *summary_keys =
            ensemble_config_alloc_keylist_from_impl_type(ensemble_config,
                                                         impltype);
        int length = stringlist_get_size(summary_keys);

        int num_elements = 0;
        for (int i = 0; i < length; ++i) {
            const char *key = stringlist_iget(summary_keys, i);

            enkf_plot_data_load(plot_data, default_fs, key, nullptr);

            num_elements += enkf_plot_data_get_size(plot_data);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto tme =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                .count();
        printf("Loaded %d elements over %d summary-keys in %.1f ms (%.1f "
               "ms/key)\n",
               num_elements, length, (double)tme, (double)tme / length);
        enkf_plot_data_free(plot_data);
        enkf_fs_decref(default_fs);

        enkf_main_free(enkf_main);
        res_config_free(res_config);
    }
    exit(0);
}
