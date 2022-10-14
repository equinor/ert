/**
 * This header contains utilities for interacting with Python via pybind11
 */

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include <ert/logging.hpp>
#include <pyerrors.h>

namespace py = pybind11;

namespace ert::detail {
struct Submodule {
    using init_type = void(py::module_);
    const char *path;
    init_type &init;

    Submodule(const char *, init_type &);
};
} // namespace ert::detail

namespace ert {
template <typename T> T *from_cwrap(py::handle obj) {
    if (obj.is_none())
        return nullptr;

    py::int_ address = obj.attr("_BaseCClass__c_pointer");
    void *pointer = PyLong_AsVoidPtr(address.ptr());

    return reinterpret_cast<T *>(pointer);
}

inline py::object to_python(void *obj) {
    auto py_obj = PyLong_FromVoidPtr(obj);
    return py::reinterpret_steal<py::object>(py_obj);
}

/**
 * Pointer container with automatic typecasting from Python
 *
 * Functions with 'Cwrap<T>' arguments will get the pointer value from the
 * Python object's '_BaseCClass__c_pointer' attribute.
 *
 * \code
 * void py_foo(Cwrap<enkf_fs_type> enkf_fs) {
 *   // enkf_fs implicitly casted from 'Cwrap<enkf_fs_type>' to 'enkf_fs_type *':
 *   enkf_fs_foo(enkf_fs);
 * }
 * \endcode
 *
 * @note No typechecking is made other than making sure that the Python type is
 *       a subclass of 'BaseCClass'.
 */
template <typename T> class Cwrap {
    T *m_ptr{};

public:
    Cwrap() = default;
    explicit Cwrap(T *ptr) : m_ptr(ptr) {}
    Cwrap(const Cwrap &) = default;
    Cwrap(Cwrap &&) = default;

    Cwrap &operator=(const Cwrap &) = default;
    Cwrap &operator=(Cwrap &&) = default;

    T *operator->() { return m_ptr; }
    const T *operator->() const { return m_ptr; }

    T &operator*() { return *m_ptr; }
    const T &operator*() const { return *m_ptr; }

    /** Implicit cast to the underlying pointer */
    operator T *() { return m_ptr; }

    /** Implicit cast to the underlying const pointer */
    operator const T *() const { return m_ptr; }
};
} // namespace ert

namespace pybind11::detail {
/**
 * Pybind11 type caster for the Cwrap type
 *
 * @note We only implement Python to C++ conversion. C++ to Python requires
 *       knowledge of the Python class name which is not obtainable just
 *       from the C++ class name.
 */
template <typename T> struct type_caster<::ert::Cwrap<T>> {
    PYBIND11_TYPE_CASTER(::ert::Cwrap<T>, const_name("cwrap"));

    bool load(handle ref, bool) {
        this->value = ::ert::Cwrap<T>{ert::from_cwrap<T>(ref)};
        return true;
    }
};
} // namespace pybind11::detail

/**
 * Define a submodule path within the Python package 'ert._clib'
 *
 * This is macro is similar to Pybind11's PYBIND11_MODULE macro. The first
 * argument is the Python submodule path, and the second is the name of the
 * py::module_ parameter (eg. 'm').
 *
 * For example, the following will create the Python module 'ert._clib.foo.bar'
 * which contains an object named 'baz' whose value is the string 'quz'.
 *
 *     ERT_CLIB_SUBMODULE("foo.bar", m) {
 *         m.add_object("baz", py::str{"quz"});
 *     }
 *
 * Multiple initialisation functions can be defined for the same submodule.
 * However, the order in which each function is called is undefined.
 *
 * Note: The name of this macro should reflect the module path of this
 * library. At the moment it is 'ert._clib', so the macro is prefixed with
 * ERT_CLIB.
 */
#define ERT_CLIB_SUBMODULE(_Path, _ModuleParam)                                \
    static void _python_submodule_init(py::module_);                           \
    static ::ert::detail::Submodule _python_submodule{_Path,                   \
                                                      _python_submodule_init}; \
    void _python_submodule_init(py::module_ _ModuleParam)

using ert::Cwrap;
