#include <ert/logging.hpp>
#include <ert/python.hpp>
#include <stdexcept>

static auto logger = ert::get_logger("enkf");

namespace fs = std::filesystem;

namespace enkf_main {
/**
  @brief Substitutes the sampled parameters into the runpath.

  Handles the substitution of all sampled parameter values into parameter
  templates. E.g. for configs including `GEN_KW key template_file target_file`,
  sampled values are gotten from fs, replaced in the contents of template_file
  and written to target_file in the runpath.

  @param ens_config Where to find the nodes (e.g. `GEN_KW key template_file
    target_file` definition).
  @param export_base_name The base name of the value_export file (e.g. if
    "parameters", value export file will e.g. be "parameters.json")
  @param run_path The run__path to write the target file in.
  @param iens The realization number.
  @param fs The enkf_fs to load sampled parameters from
*/
void ecl_write(const ensemble_config_type *ens_config,
               const char *export_base_name, const char *run_path, int iens,
               enkf_fs_type *fs) {
    value_export_type *export_value =
        value_export_alloc(run_path, export_base_name);

    for (auto &key : ensemble_config_keylist_from_var_type(
             ens_config, PARAMETER + EXT_PARAMETER)) {
        enkf_node_type *enkf_node =
            enkf_node_alloc(ensemble_config_get_node(ens_config, key.c_str()));
        node_id_type node_id = {.report_step = 0, .iens = iens};

        if (enkf_node_use_forward_init(enkf_node) &&
            !enkf_node_has_data(enkf_node, fs, node_id))
            continue;
        try {
            enkf_node_load(enkf_node, fs, node_id);
        } catch (const std::out_of_range &) {
            enkf_node_free(enkf_node);
            value_export_free(export_value);
            throw pybind11::key_error(
                fmt::format("No such parameter {} in storage", key));
        }
        enkf_node_ecl_write(enkf_node, run_path, export_value, 0);
        enkf_node_free(enkf_node);
    }
    value_export(export_value);
    value_export_free(export_value);
}
} // namespace enkf_main

ERT_CLIB_SUBMODULE("enkf_main", m) {
    using namespace py::literals;
    m.def(
        "ecl_write",
        [](Cwrap<model_config_type> model_config,
           Cwrap<ensemble_config_type> ensemble_config, char *run_path,
           int iens, Cwrap<enkf_fs_type> sim_fs) {
            enkf_main::ecl_write(
                ensemble_config,
                model_config_get_gen_kw_export_name(model_config), run_path,
                iens, sim_fs);
        },
        py::arg("model_config"), py::arg("ensemble_config"),
        py::arg("run_path"), py::arg("iens"), py::arg("sim_fs"));
}
