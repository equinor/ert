#include <unordered_map>
#include <string>
#include <string_view>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <list>
#include <utility>

using namespace std::string_literals;
namespace py = pybind11;

namespace {
py::object to_cwrap_enum(const std::string &name, int value) {
    static py::dict types =
        py::module_::import("cwrap").attr("REGISTERED_TYPES");
    auto func = types[py::str(name)].attr("type_class_or_function");
    auto obj = func(value);
    return obj;
};

py::object to_cwrap_class(const char *name, void *value) {
    py::dict types = py::module_::import("cwrap").attr("REGISTERED_TYPES");
    auto func = types[name].attr("type_class_or_function");
    auto obj = func(reinterpret_cast<std::uintptr_t>(value));
    return obj;
}

std::unordered_map<std::string, py::object> prototypes;

py::object get_prototype(const std::string &prototype, bool) {
    static const std::regex re{
        "[a-zA-Z][a-zA-Z0-9_*]*[ ]+([a-zA-Z]\\w*)[ ]*\\(:?[a-zA-Z0-9_*, ]*\\)"};
    std::smatch match;

    if (!std::regex_search(prototype, match, re))
        throw py::attribute_error("Illegal prototype definition: "s +
                                  prototype);

    auto it = prototypes.find(match[1]);
    if (it == prototypes.end())
        throw py::key_error(match[1]);
    return it->second;
}

template <typename> struct wrap_in_type { using type = py::object; };
template <typename T> using wrap_in_t = typename wrap_in_type<T>::type;

template <> struct wrap_in_type<void *> {
    using type = std::optional<py::int_>;
};

template <typename> struct wrap_t;

template <> struct wrap_t<void *> {
    static py::int_ to_py(void *ptr, bool) {
        return reinterpret_cast<std::uintptr_t>(ptr);
    }

    static void *from_py(std::optional<py::int_> object) {
        if (object.has_value())
            return PyLong_AsVoidPtr(object->ptr());
        return nullptr;
    }
};

#define WRAP_TYPE_PYBIND11(_C, _Py)                                            \
    template <> struct wrap_t<_C> {                                            \
        static _Py to_py(_C value, bool) { return value; }                     \
        static _C from_py(_Py value) { return value; }                         \
    };

WRAP_TYPE_PYBIND11(int, py::int_);
WRAP_TYPE_PYBIND11(long, py::int_);
WRAP_TYPE_PYBIND11(double, py::float_);
WRAP_TYPE_PYBIND11(bool, py::bool_);

template <> struct wrap_t<const char *> {
    static py::str to_py(const char *value, bool) { return value; }

    static const char *from_py(py::object object) {
        auto x = new std::string{PyUnicode_AsUTF8(object.ptr())};
        return x->c_str();
    }
};

template <> struct wrap_t<const float *> {
    static const float *from_py(py::object object) {
        throw std::runtime_error{"Not implemented"};
    }
};

template <> struct wrap_t<const double *> {
    static const double *from_py(py::object object) {
        throw std::runtime_error{"Not implemented"};
    }
};

template <> struct wrap_t<FILE *> {
    static FILE *from_py(py::object object) {
        throw std::runtime_error{"Not implemented"};
    }
};
} // namespace

#define WRAP_TYPE_UNSAFE(_Name)                                                \
    typedef struct _Name##_struct _Name##_type;                                \
    namespace {                                                                \
    template <> struct wrap_t<_Name##_type *> {                                \
        using type = _Name##_type *;                                           \
                                                                               \
        static py::object to_py(type obj, bool ref) {                          \
            return to_cwrap_class(ref ? #_Name "_ref" : #_Name "_obj", obj);   \
        }                                                                      \
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
        static py::object to_py(_Name obj, bool) {                             \
            return to_cwrap_enum(#_Name, obj);                                 \
        }                                                                      \
        static _Name from_py(py::object obj) {                                 \
            return static_cast<_Name>(py::cast<int>(obj.attr("value")));       \
        }                                                                      \
    };                                                                         \
    }

namespace {
class bound_cpp_function : public py::cpp_function {
public:
    using py::cpp_function::cpp_function;

    void into_instance_method() {
        auto *func = reinterpret_cast<PyCFunctionObject *>(m_ptr);
        m_ptr = PyInstanceMethod_New(m_ptr);
        if (!m_ptr)
            py::pybind11_fail("cpp_function::cpp_function(): Could not "
                              "allocate instance method object");
        Py_DECREF(func);
    }
};

template <typename T, typename... Args>
void register_function(const char *name, T (*func)(Args...), bool bind,
                       bool ref) {
    auto caster = [func, ref](wrap_in_t<Args>... args) -> py::object {
        if constexpr (std::is_same_v<T, void>) {
            func(wrap_t<Args>::from_py(args)...);
            return py::none{};
        } else {
            auto t = func(wrap_t<Args>::from_py(args)...);
            return wrap_t<T>::to_py(t, ref);
        }
    };

    auto cpp_function = bound_cpp_function{caster};
    if (bind)
        cpp_function.into_instance_method();

    prototypes.insert({name, cpp_function});
}

class Export {
public:
    template <typename T, typename... Args>
    Export(const char *name, T (*f)(Args...), bool bind, bool ref) {
        register_function(name, f, bind, ref);
    }
};

} // namespace

#define EXPORT_UNBOUND(_Name, _T, ...)                                         \
    extern "C" _T _Name(__VA_ARGS__);                                          \
    namespace {                                                                \
    Export _export_##_Name{#_Name, _Name, false, false};                       \
    }

#define EXPORT_REF_UNBOUND(_Name, _T, ...)                                     \
    extern "C" _T _Name(__VA_ARGS__);                                          \
    namespace {                                                                \
    Export _export_##_Name{#_Name, _Name, false, true};                        \
    }

#define EXPORT(_Name, _T, ...)                                                 \
    extern "C" _T _Name(__VA_ARGS__);                                          \
    namespace {                                                                \
    Export _export_##_Name{#_Name, _Name, true, false};                        \
    }

#define EXPORT_REF(_Name, _T, ...)                                             \
    extern "C" _T _Name(__VA_ARGS__);                                          \
    namespace {                                                                \
    Export _export_##_Name{#_Name, _Name, true, true};                         \
    }

WRAP_TYPE_UNSAFE(bool_vector)
WRAP_TYPE_UNSAFE(int_vector)
WRAP_TYPE_UNSAFE(double_vector)
WRAP_TYPE_UNSAFE(string_hash)
WRAP_TYPE_UNSAFE(matrix)
WRAP_TYPE_UNSAFE(hash)
WRAP_TYPE_UNSAFE(node_id)

WRAP_TYPE_UNSAFE(ecl_data_type)
WRAP_TYPE_UNSAFE(ecl_file)
WRAP_TYPE_UNSAFE(ecl_file_view)
WRAP_TYPE_UNSAFE(ecl_kw)
WRAP_TYPE_UNSAFE(ecl_rsthead)
WRAP_TYPE_UNSAFE(fortio)
WRAP_TYPE_UNSAFE(ecl_grav)
WRAP_TYPE_UNSAFE(ecl_subsidence)
WRAP_TYPE_UNSAFE(ecl_grid)
WRAP_TYPE_UNSAFE(ecl_region)
WRAP_TYPE_UNSAFE(fault_block)
WRAP_TYPE_UNSAFE(fault_block_collection)
WRAP_TYPE_UNSAFE(fault_block_layer)
WRAP_TYPE_UNSAFE(layer)
WRAP_TYPE_UNSAFE(ecl_rft)
WRAP_TYPE_UNSAFE(ecl_rft_file)
WRAP_TYPE_UNSAFE(rft_cell)
WRAP_TYPE_UNSAFE(smspec_node)
WRAP_TYPE_UNSAFE(ecl_sum)
WRAP_TYPE_UNSAFE(ecl_sum_vector)
WRAP_TYPE_UNSAFE(ecl_sum_tstep)
WRAP_TYPE_UNSAFE(geo_polygon)
WRAP_TYPE_UNSAFE(geo_polygon_collection)
WRAP_TYPE_UNSAFE(geo_pointset)
WRAP_TYPE_UNSAFE(geo_region)
WRAP_TYPE_UNSAFE(surface)
WRAP_TYPE_UNSAFE(ert_test)
WRAP_TYPE_UNSAFE(arg_pack)
WRAP_TYPE_UNSAFE(thread_pool)
WRAP_TYPE_UNSAFE(permutation_vector)
WRAP_TYPE_UNSAFE(rng)
WRAP_TYPE_UNSAFE(stringlist)
WRAP_TYPE_UNSAFE(well_connection)
WRAP_TYPE_UNSAFE(well_info)
WRAP_TYPE_UNSAFE(well_segment)
WRAP_TYPE_UNSAFE(well_state)
WRAP_TYPE_UNSAFE(well_time_line)

WRAP_TYPE_ENUM(config_unrecognized_enum);
WRAP_TYPE_ENUM(analysis_module_options_enum);
WRAP_TYPE_ENUM(ert_impl_type_enum);
WRAP_TYPE_ENUM(config_content_type_enum);
WRAP_TYPE_ENUM(enkf_run_mode_enum);
WRAP_TYPE_ENUM(active_mode_enum);
WRAP_TYPE_ENUM(enkf_var_type_enum);
WRAP_TYPE_ENUM(queue_driver_enum);
WRAP_TYPE_ENUM(history_source_enum);
WRAP_TYPE_ENUM(enkf_field_file_format_enum);
WRAP_TYPE_ENUM(job_status_type_enum);
WRAP_TYPE_ENUM(enkf_truncation_type_enum);
WRAP_TYPE_ENUM(realisation_state_enum);
WRAP_TYPE_ENUM(message_level_enum);
WRAP_TYPE_ENUM(field_type_enum);
WRAP_TYPE_ENUM(enkf_init_mode_enum);
WRAP_TYPE_ENUM(rng_alg_type_enum);
WRAP_TYPE_ENUM(hook_runtime_enum);
WRAP_TYPE_ENUM(job_submit_status_type_enum);
WRAP_TYPE_ENUM(enkf_obs_impl_type);

WRAP_TYPE_ENUM(load_fail_type);
WRAP_TYPE_ENUM(gen_data_file_format_type);
WRAP_TYPE_ENUM(ui_return_status);
WRAP_TYPE_ENUM(enkf_fs_type_enum);

EXPORT_UNBOUND(enkf_defaults_get_default_gen_kw_export_name, const char *);

#include "wrap_types-inl.hpp"
#include "exports-inl.hpp"

EXPORT(config_schema_item_valid_string, config_content_type_enum, const char *,
       bool);
// EXPORT(util_sscanf_bool, bool, const char*, bool*);

#define CONFIG_GET(_Key) EXPORT_UNBOUND(config_keys_get_##_Key, const char *);

CONFIG_GET(config_directory_key);
CONFIG_GET(config_file_key);
CONFIG_GET(queue_system_key);
CONFIG_GET(run_template_key);
CONFIG_GET(gen_kw_key);
CONFIG_GET(queue_option_key);
CONFIG_GET(install_job_key);
CONFIG_GET(refcase_key);
CONFIG_GET(install_job_directory_key);
CONFIG_GET(umask_key);
CONFIG_GET(gen_kw_export_name_key);
CONFIG_GET(hook_workflow_key);
CONFIG_GET(template_key);
CONFIG_GET(log_file_key);
CONFIG_GET(log_level_key);
CONFIG_GET(update_log_path_key);
CONFIG_GET(summary_key);
CONFIG_GET(max_runtime_key);
CONFIG_GET(min_realizations_key);
CONFIG_GET(max_submit_key);
CONFIG_GET(data_kw_key);
CONFIG_GET(runpath_file_key);
CONFIG_GET(eclbase_key);
CONFIG_GET(data_file_key);
CONFIG_GET(grid_key);
CONFIG_GET(refcase_list_key);
CONFIG_GET(end_date_key);
CONFIG_GET(schedule_prediction_file_key);
CONFIG_GET(num_realizations_key);
CONFIG_GET(enspath_key);
CONFIG_GET(history_source_key);
CONFIG_GET(obs_config_key);
CONFIG_GET(time_map_key);
CONFIG_GET(jobname_key);
CONFIG_GET(forward_model_key);
CONFIG_GET(simulation_job_key);
CONFIG_GET(max_resample_key);
CONFIG_GET(data_root_key);
CONFIG_GET(rftpath_key);
CONFIG_GET(runpath_key);
CONFIG_GET(gen_data_key);
CONFIG_GET(result_file);
CONFIG_GET(report_steps);
CONFIG_GET(input_format);
CONFIG_GET(ecl_file);
CONFIG_GET(output_format);
CONFIG_GET(init_files);
CONFIG_GET(random_seed);
CONFIG_GET(license_path_key);
CONFIG_GET(setenv_key);
CONFIG_GET(job_script_key);
CONFIG_GET(num_cpu_key);
CONFIG_GET(define_key);
CONFIG_GET(load_workflow_job_key);
CONFIG_GET(workflow_job_directory_key);
CONFIG_GET(load_workflow_key);
CONFIG_GET(iter_case_key);
CONFIG_GET(iter_count_key);
CONFIG_GET(iter_retry_count_key);
CONFIG_GET(alpha);
CONFIG_GET(std_cutoff);
CONFIG_GET(stop_long_running);
CONFIG_GET(single_node_update);
CONFIG_GET(rerun);
CONFIG_GET(rerun_start);
CONFIG_GET(analysis_copy);
CONFIG_GET(analysis_select);
CONFIG_GET(analysis_set_var);
CONFIG_GET(slurm_sbatch_option);
CONFIG_GET(slurm_scancel_option);
CONFIG_GET(slurm_scontrol_option);
CONFIG_GET(slurm_squeue_option);
CONFIG_GET(slurm_partition_option);
CONFIG_GET(slurm_squeue_timeout_option);
CONFIG_GET(slurm_max_runtime_option);
CONFIG_GET(slurm_memory_option);
CONFIG_GET(slurm_memory_per_cpu_option);
CONFIG_GET(slurm_exclude_host_option);
CONFIG_GET(slurm_include_host_option);

CONFIG_GET(gen_param_key);
CONFIG_GET(forward_init_key);
CONFIG_GET(min_std_key);
CONFIG_GET(key_key);
CONFIG_GET(kw_tag_format_key);
CONFIG_GET(surface_key);
CONFIG_GET(base_surface_key);
CONFIG_GET(field_key);
CONFIG_GET(init_transform_key);
CONFIG_GET(input_transform_key);
CONFIG_GET(output_transform_key);
CONFIG_GET(min_key);
CONFIG_GET(max_key);
CONFIG_GET(parameter_key);
CONFIG_GET(general_key);
CONFIG_GET(pred_key);
CONFIG_GET(container_key);

void init_exports(py::module_ m) {
    m.def("ResPrototype", &get_prototype, py::arg{"prototype"},
          py::arg{"bind"} = true);
}
