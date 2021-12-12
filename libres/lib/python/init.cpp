#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ert/enkf/obs_vector.hpp>
#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/ensemble_config.hpp>

namespace py = pybind11;

obs_vector_type *to_obs_vector_type(py::object obj) {
    py::object address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());
    return obs_vector_safe_cast(pointer);
}

analysis_config_type *to_analysis_config_type(py::object obj) {
    static py::object class_ =
        py::module_::import("res.enkf.analysis_config").attr("AnalysisConfig");
    if (!py::isinstance(obj, class_))
        throw py::type_error("Wrong type my friend");

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());
    return reinterpret_cast<analysis_config_type *>(pointer);
}

PYBIND11_MODULE(_lib, m) {
    m.def(
        "obs_vector_get_step_list",
        [](py::object self) {
            auto obs_vector = to_obs_vector_type(self);
            return obs_vector_get_step_list(obs_vector);
        },
        py::arg("self"));
    m.def(
        "analysis_config_module_names",
        [](py::object self) {
            auto analysis_config = to_analysis_config_type(self);
            return analysis_config_module_names(analysis_config);
        },
        py::arg("self"));

    void set_site_config(const std::string &);
    m.def("set_site_config", &set_site_config, py::arg{"site_config"});

    void init_exports(py::module_);
    init_exports(m.def_submodule("exports"));
}
