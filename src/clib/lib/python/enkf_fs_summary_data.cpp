#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/ensemble_config.hpp>

#include <ert/python.hpp>

ERT_CLIB_SUBMODULE("enkf_fs_summary_data", m) {
    m.def(
        "get_summary_data",
        [](Cwrap<enkf_fs_type> fs, const std::vector<std::string> &summary_keys,
           const std::vector<int> &realizations, const int time_map_size) {
            py::array_t<double, 2> array(
                {time_map_size * realizations.size(), summary_keys.size()});
            auto data = array.mutable_unchecked();
            int summary_key_index = 0;
            for (const auto &key : summary_keys) {
                auto config_node =
                    enkf_config_node_alloc_summary(key.c_str(), LOAD_FAIL_WARN);
                enkf_node_type *work_node = enkf_node_alloc(config_node);

                for (size_t iens_index{}; iens_index < realizations.size();
                     ++iens_index) {
                    auto iens = realizations[iens_index];
                    auto summary_vector =
                        enkf_node_user_get_vector(work_node, fs, iens);
                    for (int index = 1; index <= time_map_size; ++index) {
                        double value = (index < summary_vector.size())
                                           ? summary_vector[index]
                                           : NAN;
                        data(iens_index * time_map_size + (index - 1),
                             summary_key_index) = value;
                    }
                }
                summary_key_index++;
                enkf_node_free(work_node);
                enkf_config_node_free(config_node);
            }
            return array;
        },
        py::arg("fs"), py::arg("summary_keys"), py::arg("realizations"),
        py::arg("time_map_size"));
}
