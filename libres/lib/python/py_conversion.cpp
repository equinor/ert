/*
   When using pybind11 with cwraped objects we need to convert them from the
   python class to the corresponding c class. This is a collection of functions
   to help do that conversion.
*/

#include <ert/enkf/enkf_main.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace py_conversion {

enkf_main_type *to_enkf_main_type(py::object obj) {
    static py::object class_ =
        py::module_::import("res.enkf.enkf_fs_manager").attr("EnkfFsManager");
    if (!py::isinstance(obj, class_))
        throw py::type_error("Not of type EnKFMain");

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());

    return reinterpret_cast<enkf_main_type *>(pointer);
}

ert_run_context_type *to_run_context_type(py::object obj) {
    static py::object class_ =
        py::module_::import("res.enkf.ert_run_context").attr("ErtRunContext");
    if (!py::isinstance(obj, class_))
        throw py::type_error("Not of type ErtRunContext");

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());

    return reinterpret_cast<ert_run_context_type *>(pointer);
}

enkf_fs_type *to_enkf_fs_type(py::object obj) {
    static py::object class_ =
        py::module_::import("res.enkf.enkf_fs").attr("EnkfFs");
    if (!py::isinstance(obj, class_))
        throw py::type_error("Not of type EnkfFs");

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());

    return reinterpret_cast<enkf_fs_type *>(pointer);
}

ensemble_config_type *to_ensemble_config_type(py::object obj) {
    static py::object class_ =
        py::module_::import("res.enkf.ensemble_config").attr("EnsembleConfig");
    if (!py::isinstance(obj, class_))
        throw py::type_error("Not of type EnsembleConfig");

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());

    return reinterpret_cast<ensemble_config_type *>(pointer);
}

} // namespace py_conversion
