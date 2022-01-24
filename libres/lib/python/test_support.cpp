/**
 * This file contains supporting functions for the Python libres tests.
 */
#include "ert/enkf/local_ministep.hpp"
#include <ert/python.hpp>

namespace {
template <typename T> T *from_cwrap(py::handle obj, const char *class_name) {
    py::str name = obj.attr("__class__").attr("__name__");
    std::string sname{name};
    if (sname != class_name)
        throw py::attribute_error(
            fmt::format("Expected class '{}', got '{}'", class_name, sname));

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    return reinterpret_cast<T *>(PyLong_AsVoidPtr(address.ptr()));
}

void add_active_indices(py::handle obj, const std::string &key,
                        const std::vector<int> &indices) {
    auto ministep = from_cwrap<local_ministep_type>(obj, "LocalMinistep");

    auto &active_list = ministep->get_active_data_list(key.c_str());
    for (auto index : indices) {
        active_list.add_index(index);
    }
}
} // namespace

RES_LIB_SUBMODULE("test_support", m) {
    using namespace py::literals;

    /* Used in 'res/enkf/test_es_update::test_localization' */
    m.def("local_ministep_activate_indices", &add_active_indices, "self"_a,
          "key"_a, "indices"_a);
}
