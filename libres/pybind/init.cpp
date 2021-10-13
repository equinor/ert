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

    double get_value() const { return summary_obs_get_value(m_ptr.get()); }

    double get_std() const { return summary_obs_get_std(m_ptr.get()); }

    double get_std_scaling() const {
        return summary_obs_get_std_scaling(m_ptr.get());
    }

    std::string get_summary_key() const {
        return summary_obs_get_summary_key(m_ptr.get());
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
        .def("_get_value", &summary_observation::get_value)
        .def("_get_std", &summary_observation::get_std)
        .def("_get_std_scaling", &summary_observation::get_std_scaling)
        .def("_get_summary_key", &summary_observation::get_summary_key)
        .def("_update_std_scale", &summary_observation::update_std_scale)
        .def("_set_std_scale", &summary_observation::set_std_scale);
}
