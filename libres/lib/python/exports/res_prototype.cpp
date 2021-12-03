#include <unordered_map>
#include <string>
#include <string_view>
#include <pybind11/pybind11.h>
#include <regex>
#include <list>
#include <utility>

using namespace std::string_literals;
using namespace std::string_view_literals;
namespace py = pybind11;

namespace {
struct dummy_t{};
struct IExport;
py::object to_cwrap_enum(const std::string &name, int value) {
    static py::dict types =
        py::module_::import("cwrap").attr("REGISTERED_TYPES");
    auto func = types[py::str(name)].attr("type_class_or_function");
    return func(value);
};

std::unordered_map<std::string, py::cpp_function> prototypes;
std::list<IExport*> exports;

py::handle get_prototype(py::object cls, const std::string &prototype,
                         bool bind) {
    static const std::regex re{
        "[a-zA-Z][a-zA-Z0-9_*]*[ ]+([a-zA-Z]\\w*)[ ]*\\(:?[a-zA-Z0-9_*, ]*\\)"};
    std::smatch match;

    if (!std::regex_search(prototype, match, re))
        throw py::attribute_error("Illegal prototype definition: "s +
                                  prototype);
    return prototypes.at(match[1]);
}

template <typename> struct wrap_t;

template <> struct wrap_t<void *> {
    static py::object to_py(void *ptr) {
        return py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(ptr));
    }

    static void *from_py(py::object obj) { return PyLong_AsVoidPtr(obj.ptr()); }
};

template <typename> using to_py_object_t = py::object;

struct IExport {
    IExport() {
        exports.push_back(this);
    }
    virtual ~IExport() {}
    virtual void add(py::class_<dummy_t> &) = 0;
};

template <typename RetT, typename... ArgsT> class ExportUnbound : public IExport {
    using type = RetT(ArgsT...);
    std::string m_name;
    type *m_func;

public:
    ExportUnbound(const std::string &name, type &func)
        : m_name(name), m_func(&func) {}

    void add(py::class_<dummy_t> &m) override {
        auto func = m_func;

        auto py_func = [func](to_py_object_t<ArgsT>... args) -> py::object {
            py::gil_scoped_release remove_gil;
            if constexpr (std::is_same_v<RetT, void>) {
                func(wrap_t<ArgsT>::from_py(args)...);
                return py::none{};
            } else {
                return wrap_t<RetT>::to_py(
                    func(wrap_t<ArgsT>::from_py(args)...));
            }
        };
        prototypes.emplace(m_name, py_func);
    }
};

template <typename RetT, typename... ArgsT> class ExportBound : public IExport {
    using type = RetT(ArgsT...);
    std::string m_name;
    type *m_func;

public:
    ExportBound(const std::string &name, type &func)
        : m_name(name), m_func(&func) {}

    void add(py::class_<dummy_t> &m) override {
        auto func = m_func;

        auto py_func = [func](py::object self, py::args) -> py::object {
            py::gil_scoped_release remove_gil;
            if constexpr (std::is_same_v<RetT, void>) {
                func(wrap_t<ArgsT>::from_py(args)...);
                return py::none{};
            } else {
                return wrap_t<RetT>::to_py(
                    func(wrap_t<ArgsT>::from_py(args)...));
            }
        };

        auto cpp_func = py::cpp_function{py_func, py::arg{"self"}, py::arg{"args"}};
        prototypes.emplace(m_name, cpp_func);
    }
};
} // namespace

#define S(x) #x

#define EXPORT_UNBOUND(_RetT, _Name, ...)                                      \
    extern "C" _RetT _Name(__VA_ARGS__);                                       \
    namespace {                                                                \
    ExportUnbound _export_##_Name{#_Name, _Name};                              \
    }

#define EXPORT(_RetT, _Type, _Name, ...)                                       \
    extern "C" _RetT _Type##_##_Name(_Type##_type *, ##__VA_ARGS__);           \
    namespace {                                                                \
    ExportBound _export_##_Type##_##_Name{S(_Type##_##_Name),                  \
                                          _Type##_##_Name};                    \
    }

#define WRAP_TYPE_PYBIND11(_CType, _PyType)                                    \
    namespace {                                                                \
    template <> struct wrap_t<_CType> {                                        \
        static _PyType to_py(_CType value) { return value; }                   \
        static _CType from_py(_PyType value) { return value; }                 \
    };                                                                         \
    }

#define WRAP_TYPE_UNSAFE(_Name)                                                \
    typedef struct _Name##_struct _Name##_type;                                \
    namespace {                                                                \
    template <> struct wrap_t<_Name##_type *> {                                \
        using type = _Name##_type *;                                           \
                                                                               \
        static py::object to_py(type obj) { return py::int_{0}; }              \
                                                                               \
        static type from_py(py::object obj) {                                  \
            auto addr = obj.attr("_BaseCClass__c_pointer");                    \
            auto ptr = PyLong_AsVoidPtr(addr.ptr());                           \
            return reinterpret_cast<type>(ptr);                                \
        }                                                                      \
    };                                                                         \
    }

#define WRAP_TYPE_ENUM(_Name)                                                  \
    enum _Name {};                                                             \
    namespace {                                                                \
    template <> struct wrap_t<_Name> {                                         \
        static py::object to_py(_Name obj) {                                   \
            return to_cwrap_enum(#_Name, obj);                                 \
        }                                                                      \
        static py::int_ from_py(py::object obj) { return obj.attr("value"); }  \
    };                                                                         \
    }

WRAP_TYPE_PYBIND11(int, py::int_);
WRAP_TYPE_PYBIND11(bool, py::bool_);
WRAP_TYPE_UNSAFE(active_list);
WRAP_TYPE_ENUM(active_mode_enum);

EXPORT_UNBOUND(void *, active_list_alloc);
EXPORT(void, active_list, free);
EXPORT(void, active_list, add_index, int);
EXPORT(int, active_list, get_active_size, int);
EXPORT(active_mode_enum, active_list, get_mode);

void init_exports(py::module_ m) {
    py::class_<dummy_t>cls(m, "ResPrototype2");
        cls.def_static("__new__", &get_prototype, py::arg{"cls"},
                    py::arg{"prototype"}, py::arg{"bind"} = true);

    for (auto export_ : exports)
        export_->add(cls);
}
