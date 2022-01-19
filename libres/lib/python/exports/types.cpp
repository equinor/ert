#include "types.hpp"


py::object ert::detail::generic_enum_to_cwrap(const char *name, int value) {
    py::dict types = py::module_::import("cwrap").attr("REGISTERED_TYPES");
    auto init = types[name].attr("type_class_or_function");
    return init(value);
}

py::object ert::detail::generic_struct_to_cwrap(const char *name, const void *value) {
    py::dict types = py::module_::import("cwrap").attr("REGISTERED_TYPES");
    auto init = types[name].attr("type_class_or_function");
    auto addr = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(const_cast<void*>(value)));
    return init(value);
}

int ert::detail::generic_enum_from_cwrap(const char *name, py::object object) {
    py::int_ value = object.attr("value");
    return value;
}

void *ert::detail::generic_struct_from_cwrap(const char *name, py::object object) {
    py::int_ addr = object.attr("_BaseCClass__c_pointer");
    return PyLong_AsVoidPtr(addr.ptr());
}
