#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_plot_genvector.hpp>
#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("enkf_fs_general_data", m) {
    m.def(
        "gendata_get_realizations",
        [](Cwrap<enkf_config_node_type> config_node, Cwrap<enkf_fs_type> fs,
           const std::vector<int> &realizations, int report_step) {
            if (realizations.empty())
                return py::array_t<double, 2>();

            auto vector =
                enkf_plot_genvector_alloc(config_node, realizations.front());
            enkf_plot_genvector_load(vector, fs, report_step);
            size_t data_size = enkf_plot_genvector_get_size(vector);
            py::array_t<double, 2> array({data_size, realizations.size()});
            auto data = array.mutable_unchecked();

            for (size_t iens_index{}; iens_index < realizations.size();
                 ++iens_index) {
                if (iens_index > 0) {
                    enkf_plot_genvector_free(vector);
                    vector = enkf_plot_genvector_alloc(
                        config_node, realizations[iens_index]);
                    enkf_plot_genvector_load(vector, fs, report_step);

                    if (enkf_plot_genvector_get_size(vector) != data_size) {
                        enkf_plot_genvector_free(vector);
                        throw py::value_error("GEN_DATA vector size mismatch");
                    }
                }

                int data_index;
                for (data_index = 0; data_index < data_size; data_index++) {
                    data(data_index, iens_index) =
                        enkf_plot_genvector_iget(vector, data_index);
                }
            }
            enkf_plot_genvector_free(vector);
            return array;
        },
        py::arg("config_node"), py::arg("realizations"), py::arg("storage"),
        py::arg("report_step"));
}
