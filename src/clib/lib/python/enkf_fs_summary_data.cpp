#include <ert/enkf/enkf_plot_tvector.hpp>

#include <ert/python.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

ERT_CLIB_SUBMODULE("enkf_fs_summary_data", m) {
    m.def(
        "get_summary_data",
        [](Cwrap<ensemble_config_type> ensemble_config, Cwrap<enkf_fs_type> fs,
           const std::vector<std::string> &summary_keys,
           const std::vector<int> &realizations, const int time_map_size) {
            const int realization_size = std::size(realizations);
            const int summary_key_size = std::size(summary_keys);
            const size_t size =
                realization_size * summary_key_size * time_map_size;

            double *data = new double[size];
            std::fill_n(data, size, NAN);

            int summary_key_index = 0;
            for (const auto &key : summary_keys) {
                auto ensemble_config_node =
                    ensemble_config_get_node(ensemble_config, key.c_str());

                auto &state_map = enkf_fs_get_state_map(fs);

                std::vector<bool> mask =
                    state_map.select_matching(STATE_HAS_DATA);

                for (int iens = 0; iens < realization_size; iens++) {
                    if (mask[iens]) {
                        enkf_plot_tvector_type *vector =
                            enkf_plot_tvector_alloc(ensemble_config_node, iens);
                        enkf_plot_tvector_load(vector, fs, nullptr);
                        int realization_vector_size =
                            enkf_plot_tvector_size(vector);

                        for (int index = 1; index < realization_vector_size;
                             index++) {
                            if (enkf_plot_tvector_iget_active(vector, index)) {
                                double value =
                                    enkf_plot_tvector_iget_value(vector, index);
                                data[iens * (time_map_size * summary_key_size) +
                                     (((index - 1) * summary_key_size) +
                                      summary_key_index)] = value;
                            }
                        }
                        enkf_plot_tvector_free(vector);
                    }
                }
                summary_key_index++;
            }

            py::capsule free_when_done(data, [](void *f) {
                double *data = reinterpret_cast<double *>(f);
                delete[] data;
            });

            return py::array_t<double>(
                {time_map_size * realization_size, summary_key_size}, // shape
                {(summary_key_size) * sizeof(double),
                 sizeof(double)}, // C-style contiguous strides for double
                data,             // the data pointer
                free_when_done);  // numpy array references this parent
        },
        py::arg("ens_cfg"), py::arg("fs"), py::arg("summary_keys"),
        py::arg("realizations"), py::arg("time_map_size"));
}
