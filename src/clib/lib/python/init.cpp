#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/obs_vector.hpp>
#include <ert/python.hpp>
#include <ert/res_util/string.hpp>
#include <pyerrors.h>

namespace {
auto &submodules() {
    static std::vector<ert::detail::Submodule *> submodules;
    return submodules;
}
} // namespace

ert::detail::Submodule::Submodule(const char *path, init_type &init)
    : path(path), init(init) {
    submodules().push_back(this);
}

PYBIND11_MODULE(_clib, m) {
    py::register_exception_translator([](std::exception_ptr p) {
        if (!p)
            return;

        try {
            std::rethrow_exception(p);
        } catch (const std::ios_base::failure &e) {
            PyErr_SetString(PyExc_OSError, e.what());
            return;
        }
    });

    /* Initialise submodules */
    for (auto submodule : submodules()) {
        py::module_ node = m;
        ert::split(submodule->path, '.', [&node](auto name) {
            std::string sname{name};
            if (hasattr(node, sname.c_str())) {
                node = node.attr(sname.c_str());
            } else {
                node = node.def_submodule(sname.c_str());
            }
        });

        submodule->init(node);
    }

    m.def(
        "obs_vector_get_step_list",
        [](py::object self) {
            auto obs_vector = ert::from_cwrap<obs_vector_type>(self);
            return obs_vector_get_step_list(obs_vector);
        },
        py::arg("self"));
    m.def(
        "analysis_config_module_names",
        [](py::object self) {
            auto analysis_config = ert::from_cwrap<analysis_config_type>(self);
            return analysis_config_module_names(analysis_config);
        },
        py::arg("self"));
}
