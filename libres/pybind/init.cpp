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

namespace {

struct deleter {
    void operator()(summary_obs_type *ptr) { summary_obs_free(ptr); }
};

class summary_observation {
public:
    std::unique_ptr<summary_obs_type, deleter> m_ptr;

public:
    summary_observation(summary_obs_type *ptr) : m_ptr(ptr) {}
    summary_observation(summary_observation &&) = default;
    summary_observation &operator=(summary_observation &&) = default;

    static summary_observation alloc(const std::string &arg0,
                                     const std::string &arg1, double arg2,
                                     double arg3) {
        return {summary_obs_alloc(arg0.c_str(), arg1.c_str(), arg2, arg3)};
    }

    void update_std_scale(double arg0, py::object arg1) {
        summary_obs_update_std_scale(m_ptr.get(), arg0,
                                     cwrap_cast<active_list_type *>(arg1));
    }

    void set_std_scale(double arg0) {
        summary_obs_set_std_scale(m_ptr.get(), arg0);
    }
};
} // namespace

PYBIND11_MODULE(_clib, m) {
    py::class_<summary_observation>(m, "_SummaryObservationImpl")
        .def(py::init(&summary_observation::alloc))
        .def("_get_value",
             [](const summary_observation &self) { return self.m_ptr->value; })
        .def("_get_std",
             [](const summary_observation &self) { return self.m_ptr->std; })
        .def("_get_std_scaling",
             [](const summary_observation &self) {
                 return self.m_ptr->std_scaling;
             })
        .def("_get_summary_key",
             [](const summary_observation &self) {
                 return std::string(self.m_ptr->summary_key);
             })
        .def("_update_std_scale", &summary_observation::update_std_scale)
        .def("_set_std_scale", &summary_observation::set_std_scale);
}
