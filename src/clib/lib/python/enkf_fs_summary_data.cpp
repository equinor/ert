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
            py::array_t<double, 2> array(
                {time_map_size * realizations.size(), summary_keys.size()});
            auto data = array.mutable_unchecked();

            int summary_key_index = 0;
            for (const auto &key : summary_keys) {
                auto ensemble_config_node =
                    ensemble_config_get_node(ensemble_config, key.c_str());

                for (size_t iens_index{}; iens_index < realizations.size();
                     ++iens_index) {
                    auto iens = realizations[iens_index];
                    enkf_plot_tvector_type *vector =
                        enkf_plot_tvector_alloc(ensemble_config_node, iens);
                    enkf_plot_tvector_load(vector, fs, nullptr);

                    for (int index = 1; index < enkf_plot_tvector_size(vector);
                         index++) {
                        double value =
                            enkf_plot_tvector_iget_active(vector, index)
                                ? enkf_plot_tvector_iget_value(vector, index)
                                : NAN;
                        data(iens_index * time_map_size + (index - 1),
                             summary_key_index) = value;
                    }
                    enkf_plot_tvector_free(vector);
                }
                summary_key_index++;
            }
            return array;
        },
        py::arg("ens_cfg"), py::arg("fs"), py::arg("summary_keys"),
        py::arg("realizations"), py::arg("time_map_size"));
}
