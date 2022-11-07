#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/python.hpp>

namespace {
std::vector<double> load_gen_data(enkf_config_node_type *config_node, int iens,
                                  int report_step, enkf_fs_type *fs) {
    enkf_node_type *work_node = enkf_node_alloc(config_node);
    node_id_type node_id = {.report_step = report_step, .iens = iens};
    enkf_node_load(work_node, fs, node_id);
    const gen_data_type *node =
        (const gen_data_type *)enkf_node_value_ptr(work_node);
    std::vector<double> vector;
    vector.resize(gen_data_get_size(node));
    std::copy_n(gen_data_get_double_vector(node), gen_data_get_size(node),
                vector.data());
    enkf_node_free(work_node);
    return vector;
}
} // namespace

ERT_CLIB_SUBMODULE("enkf_fs_general_data", m) {
    m.def(
        "gendata_get_realizations",
        [](Cwrap<enkf_config_node_type> config_node, Cwrap<enkf_fs_type> fs,
           const std::vector<int> &realizations, int report_step) {
            if (realizations.empty())
                return py::array_t<double, 2>();

            auto gen_data_data = load_gen_data(
                config_node, realizations.front(), report_step, fs);
            auto data_size = gen_data_data.size();
            py::array_t<double, 2> array({data_size, realizations.size()});
            auto data = array.mutable_unchecked();

            for (size_t iens_index{}; iens_index < realizations.size();
                 ++iens_index) {
                if (iens_index > 0) {
                    gen_data_data = load_gen_data(
                        config_node, realizations[iens_index], report_step, fs);
                    if (gen_data_data.size() != data_size) {
                        throw py::value_error("GEN_DATA vector size mismatch");
                    }
                }

                for (size_t data_index{}; data_index < gen_data_data.size();
                     ++data_index)
                    data(data_index, iens_index) = gen_data_data[data_index];
            }
            return array;
        },
        py::arg("config_node"), py::arg("realizations"), py::arg("storage"),
        py::arg("report_step"));
}
