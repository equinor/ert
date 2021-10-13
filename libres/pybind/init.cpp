#include <string>
#include <memory>
#include <pybind11/pybind11.h>

#include <ert/enkf/summary_obs.hpp>

namespace py = pybind11;

template <class T> T cwrap_cast(py::object obj) {
    PyObject *handle = obj.attr("_BaseCClass__c_pointer").ptr();
    void *cwrap_ptr = PyLong_AsVoidPtr(handle);
    return reinterpret_cast<T>(cwrap_ptr);
}

using type = summary_obs_type;
using type_ref = type &;
using type_cref = const type &;
using shared = std::shared_ptr<type>;

template <class T, class B> auto fgetter(T B::*p) {
    return [p](const B &s) { return s.*p; };
}

template <class R, class T, class B> auto frgetter(T B::*p) {
    return [p](const B &s) { return R{s.*p}; };
}

PYBIND11_MODULE(_clib, m) {
    py::class_<type, shared> cls(m, "_SummaryObservationImpl");
    cls.def(py::init([](const std::string &arg0, const std::string &arg1,
                        double arg2, double arg3) {
        return shared{summary_obs_alloc(arg0.c_str(), arg1.c_str(), arg2, arg3),
                      summary_obs_free};
    }));
    cls.def("getValue", fgetter(&summary_obs_type::value));
    cls.def("getStandardDeviation", fgetter(&summary_obs_type::std));

    // This getter takes an optional index so we can't use `fgetter`
    cls.def(
        "getStdScaling",
        [](type_cref self, int index) { return self.std_scaling; },
        py::arg("index") = 0);
    cls.def("getSummaryKey", frgetter<std::string>(&summary_obs_type::summary_key));
    cls.def("updateStdScaling",
            [](type_ref self, double arg0, py::object arg1) {
                summary_obs_update_std_scale(
                    &self, arg0, cwrap_cast<active_list_type *>(arg1));
            });
    cls.def("set_std_scaling", [](type_ref self, double arg0) {
        summary_obs_set_std_scale(&self, arg0);
    });
}
