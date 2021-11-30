#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <ert/enkf/obs_vector.hpp>

namespace py = pybind11;

obs_vector_type *to_obs_vector_type(py::object obj) {
    py::object address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());
    return obs_vector_safe_cast(pointer);
}

PYBIND11_MODULE(_lib, m) {
    m.def(
        "obs_vector_get_step_list",
        [](py::object self) {
            auto obs_vector = to_obs_vector_type(self);
            return obs_vector_get_step_list(obs_vector);
        },
        py::arg("self"));
}
