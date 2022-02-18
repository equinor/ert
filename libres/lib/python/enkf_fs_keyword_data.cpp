#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/enkf_plot_gen_kw.hpp>
#include <math.h>
#include <ert/python.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

RES_LIB_SUBMODULE("enkf_fs_keyword_data", m) {
    m.def(
        "keyword_data_get_realizations",
        [](py::object config, py::object fs,
           const std::vector<std::string> &keys,
           const std::vector<int> &realizations) {
            auto ensemble_config =
                ert::from_cwrap<ensemble_config_type>(config);
            auto enkf_fs = ert::from_cwrap<enkf_fs_type>(fs);

            const int key_count = std::size(keys);
            const int realization_size = std::size(realizations);
            const size_t size = realization_size * key_count;

            double *data = new double[size];
            std::fill_n(data, size, NAN);

            for (int key_index = 0; key_index < key_count; key_index++) {

                auto key = keys.at(key_index);
                std::string keyword = "";
                auto split = key.find(":");
                if (split != std::string::npos) {
                    keyword = key.substr(split + 1);
                    key = key.substr(0, split);
                }

                bool use_log_scale = false;
                if (key.substr(0, 6) == "LOG10_") {
                    key = key.substr(6);
                    use_log_scale = true;
                }

                auto ensemble_config_node =
                    ensemble_config_get_node(ensemble_config, key.c_str());
                auto ensemble_data =
                    enkf_plot_gen_kw_alloc(ensemble_config_node);
                enkf_plot_gen_kw_load(ensemble_data, enkf_fs, true, 0, NULL);

                auto keyword_index =
                    enkf_plot_gen_kw_get_keyword_index(ensemble_data, keyword);
                for (int realization_index = 0;
                     realization_index < realization_size;
                     realization_index++) {
                    int realization = realizations.at(realization_index);
                    enkf_plot_gen_kw_vector_type *vector =
                        enkf_plot_gen_kw_iget(ensemble_data, realization);

                    auto value =
                        enkf_plot_gen_kw_vector_iget(vector, keyword_index);
                    if (use_log_scale) {
                        value = log10(value);
                    }
                    data[key_index + realization_index * key_count] = value;
                }
                enkf_plot_gen_kw_free(ensemble_data);
            }
            py::capsule free_when_done(data, [](void *f) {
                double *data = reinterpret_cast<double *>(f);
                delete[] data;
            });

            return py::array_t<double>(
                {realization_size, key_count}, // shape
                {key_count * sizeof(double),
                 sizeof(double)}, // C-style contiguous strides for double
                data,             // the data pointer
                free_when_done);  // numpy array references this parent
        },
        py::arg("config"), py::arg("fs"), py::arg("key"),
        py::arg("realizations"));
}
