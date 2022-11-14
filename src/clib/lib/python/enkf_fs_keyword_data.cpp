#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/gen_kw.hpp>
#include <ert/enkf/gen_kw_config.hpp>
#include <ert/python.hpp>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static auto logger = ert::get_logger("enkf_fs");

int get_keyword_index(const enkf_config_node_type *config_node,
                      const std::string &keyword) {
    const gen_kw_config_type *gen_kw_config =
        (const gen_kw_config_type *)enkf_config_node_get_ref(config_node);

    auto kw_count = gen_kw_config_get_data_size(gen_kw_config);
    int result = -1;
    for (int i = 0; i < kw_count; i++) {
        std::string key = gen_kw_config_iget_name(gen_kw_config, i);
        if (key == keyword) {
            result = i;
        }
    }
    return result;
}

ERT_CLIB_SUBMODULE("enkf_fs_keyword_data", m) {
    m.def(
        "keyword_data_get_realizations",
        [](Cwrap<ensemble_config_type> ensemble_config,
           Cwrap<enkf_fs_type> enkf_fs, const std::vector<std::string> &keys,
           const std::vector<int> &realizations) {
            py::array_t<double, 2> array({realizations.size(), keys.size()});
            auto data = array.mutable_unchecked();

            for (int key_index = 0; key_index < keys.size(); key_index++) {

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
                for (int realization_index = 0;
                     realization_index < realizations.size();
                     realization_index++) {
                    int realization = realizations.at(realization_index);
                    node_id_type node_id = {.report_step = 0,
                                            .iens = realization};
                    enkf_node_type *data_node =
                        enkf_node_alloc(ensemble_config_node);
                    double value = NAN;
                    if (enkf_node_try_load(data_node, enkf_fs, node_id)) {
                        const gen_kw_type *gen_kw =
                            (const gen_kw_type *)enkf_node_value_ptr(data_node);
                        auto index =
                            get_keyword_index(ensemble_config_node, keyword);
                        value = gen_kw_data_iget(gen_kw, index, true);
                        if (use_log_scale)
                            value = log10(value);
                    }
                    data(realization_index, key_index) = value;
                    enkf_node_free(data_node);
                }
            }
            return array;
        },
        py::arg("config"), py::arg("fs"), py::arg("key"),
        py::arg("realizations"));
}
