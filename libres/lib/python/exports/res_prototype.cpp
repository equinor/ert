#include <unordered_map>
#include <string>
#include <string_view>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <list>
#include <utility>
#include <type_traits>

#include "types.hpp"
#include "funcs.hpp"

using namespace std::string_literals;
namespace {
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

template <typename T> using wrap_in_t = py::object;

template <typename T> struct wrap_t {
    // statis_assert(std::is_enum_v<T>, "T must be an enum class");

    static py::object to_py(T val, bool) {
        return ert::to_cwrap<T>(val);
    }

    static T from_py(py::object obj) {
        return ert::from_cwrap<T>(obj);
    }
};

template <typename T> struct wrap_t<T*> {
    static py::object to_py(const T *ptr, bool is_ref) {
        return is_ref ? ert::to_cwrap_ref<T>(ptr) : ert::to_cwrap<T>(ptr);
    }

    static T* from_py(py::object obj) {
        return reinterpret_cast<T*>(ert::from_cwrap<T>(obj));
    }
};

template <> struct wrap_t<void *> {
    static py::object to_py(void *ptr, bool) {
        return py::reinterpret_steal<py::int_>(PyLong_FromVoidPtr(ptr));
    }

    static void* from_py(py::object obj) {
        return PyLong_AsVoidPtr(obj.ptr());
    }
};

template <> struct wrap_t<FILE*> {
    static FILE* from_py(py::object obj) {
        return reinterpret_cast<FILE*>(ert::detail::generic_struct_from_cwrap("FILE", obj));
    }
};

template <> struct wrap_t<const char *> {
    static py::str to_py(const char *str, bool) {
        return str;
    }

    static const char *from_py(py::str obj) {
        auto y = new std::string{static_cast<std::string>(obj)};
        return y->c_str();
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

} // namespace

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

const int unbound = 1;
const int return_ref = 2;

template <typename T, typename... Args>
void add_function(const char *name, T (*func)(Args...), int flags = 0) {
    bool ref = flags & return_ref;

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
    if (!(flags & unbound))
        cpp_function.into_instance_method();

    prototypes.insert({name, cpp_function});
}

} // namespace

void init_exports(py::module_ m) {
    m.def("ResPrototype", &get_prototype, py::arg{"prototype"},
          py::arg{"bind"} = true);

    m.add_object("_cleanup", py::capsule([] {
        prototypes.clear();
    }));

    // AnalysisModule
    add_function("analysis_module_alloc", analysis_module_alloc, unbound);
    add_function("analysis_module_free", analysis_module_free);
    add_function("analysis_module_set_var", analysis_module_set_var);
    add_function("analysis_module_get_table_name",
                 analysis_module_get_table_name);
    add_function("analysis_module_get_name", analysis_module_get_name);
    add_function("analysis_module_check_option", analysis_module_check_option);
    add_function("analysis_module_has_var", analysis_module_has_var);
    add_function("analysis_module_get_double", analysis_module_get_double);
    add_function("analysis_module_get_int", analysis_module_get_int);
    add_function("analysis_module_get_bool", analysis_module_get_bool);
    add_function("analysis_module_get_ptr", analysis_module_get_ptr);
    add_function("analysis_module_init_update", analysis_module_init_update);
    add_function("analysis_module_updateA", analysis_module_updateA);
    add_function("analysis_module_initX", analysis_module_initX);
    // ConfigError
    add_function("config_error_free", config_error_free);
    add_function("config_error_count", config_error_count);
    add_function("config_error_iget", config_error_iget);
    // ConfigPathElm
    add_function("config_path_elm_free", config_path_elm_free);
    add_function("config_path_elm_get_relpath", config_path_elm_get_relpath);
    add_function("config_path_elm_get_abspath", config_path_elm_get_abspath);
    // ContentNode
    add_function("config_content_node_iget", config_content_node_iget);
    add_function("config_content_node_get_size", config_content_node_get_size);
    add_function("config_content_node_get_full_string",
                 config_content_node_get_full_string);
    add_function("config_content_node_iget_type",
                 config_content_node_iget_type);
    add_function("config_content_node_iget_as_abspath",
                 config_content_node_iget_as_abspath);
    add_function("config_content_node_iget_as_relpath",
                 config_content_node_iget_as_relpath);
    add_function("config_content_node_iget_as_int",
                 config_content_node_iget_as_int);
    add_function("config_content_node_iget_as_double",
                 config_content_node_iget_as_double);
    add_function("config_content_node_iget_as_path",
                 config_content_node_iget_as_path);
    add_function("config_content_node_iget_as_bool",
                 config_content_node_iget_as_bool);
    add_function("config_content_node_iget_as_isodate",
                 config_content_node_iget_as_isodate);
    // ContentItem
    add_function("config_content_item_alloc", config_content_item_alloc,
                 unbound);
    add_function("config_content_item_get_size", config_content_item_get_size);
    add_function("config_content_item_iget_node", config_content_item_iget_node,
                 return_ref);
    add_function("config_content_item_free", config_content_item_free);
    // ContentTypeEnum
    add_function("config_schema_item_valid_string", config_schema_item_valid_string);
    // ConfigContent
    add_function("config_content_alloc", config_content_alloc, unbound);
    add_function("config_content_free", config_content_free);
    add_function("config_content_is_valid", config_content_is_valid);
    add_function("config_content_has_item", config_content_has_item);
    add_function("config_content_get_item", config_content_get_item,
                 return_ref);
    add_function("config_content_get_errors", config_content_get_errors,
                 return_ref);
    add_function("config_content_get_warnings", config_content_get_warnings,
                 return_ref);
    add_function("config_content_get_config_path",
                 config_content_get_config_path);
    add_function("config_content_add_path_elm", config_content_add_path_elm,
                 return_ref);
    add_function("config_content_add_define", config_content_add_define);
    add_function("config_content_get_size", config_content_get_size);
    add_function("config_content_alloc_keys", config_content_alloc_keys);
    // ConfigParser
    add_function("config_alloc", config_alloc, unbound);
    add_function("config_add_schema_item", config_add_schema_item, return_ref);
    add_function("config_free", config_free);
    add_function("config_parse", config_parse);
    add_function("config_get_schema_size", config_get_schema_size);
    add_function("config_get_schema_item", config_get_schema_item, return_ref);
    add_function("config_has_schema_item", config_has_schema_item);
    add_function("config_parser_add_key_values", config_parser_add_key_values);
    add_function("config_validate", config_validate);
    // ConfigSettings
    add_function("config_settings_alloc", config_settings_alloc, unbound);
    add_function("config_settings_free", config_settings_free);
    add_function("config_settings_add_setting", config_settings_add_setting);
    add_function("config_settings_add_double_setting",
                 config_settings_add_double_setting);
    add_function("config_settings_add_int_setting",
                 config_settings_add_int_setting);
    add_function("config_settings_add_string_setting",
                 config_settings_add_string_setting);
    add_function("config_settings_add_bool_setting",
                 config_settings_add_bool_setting);
    add_function("config_settings_has_key", config_settings_has_key);
    add_function("config_settings_get_value_type",
                 config_settings_get_value_type);
    add_function("config_settings_init_parser", config_settings_init_parser);
    add_function("config_settings_apply", config_settings_apply);
    add_function("config_settings_alloc_keys", config_settings_alloc_keys);
    add_function("config_settings_get_value", config_settings_get_value);
    add_function("config_settings_get_int_value",
                 config_settings_get_int_value);
    add_function("config_settings_get_double_value",
                 config_settings_get_double_value);
    add_function("config_settings_get_bool_value",
                 config_settings_get_bool_value);
    add_function("config_settings_set_value", config_settings_set_value);
    add_function("config_settings_set_int_value",
                 config_settings_set_int_value);
    add_function("config_settings_set_double_value",
                 config_settings_set_double_value);
    add_function("config_settings_set_bool_value",
                 config_settings_set_bool_value);
    // SchemaItem
    add_function("config_schema_item_alloc", config_schema_item_alloc, unbound);
    add_function("config_schema_item_free", config_schema_item_free);
    add_function("config_schema_item_iget_type", config_schema_item_iget_type);
    add_function("config_schema_item_iset_type", config_schema_item_iset_type);
    add_function("config_schema_item_set_argc_minmax",
                 config_schema_item_set_argc_minmax);
    add_function("config_schema_item_add_indexed_alternative",
                 config_schema_item_add_indexed_alternative);
    add_function("config_schema_item_set_deprecated",
                 config_schema_item_set_deprecated);
    // ErtTemplate
    add_function("ert_template_free", ert_template_free);
    add_function("ert_template_get_template_file",
                 ert_template_get_template_file);
    add_function("ert_template_get_target_file", ert_template_get_target_file);
    add_function("ert_template_get_arg_list", ert_template_get_arg_list,
                 return_ref);
    // AnalysisConfig
    add_function("analysis_config_alloc", analysis_config_alloc, unbound);
    add_function("analysis_config_alloc_load", analysis_config_alloc_load,
                 unbound);
    add_function("analysis_config_alloc_full", analysis_config_alloc_full,
                 unbound);
    add_function("analysis_config_add_module_copy",
                 analysis_config_add_module_copy);
    add_function("analysis_config_free", analysis_config_free);
    add_function("analysis_config_get_rerun", analysis_config_get_rerun);
    add_function("analysis_config_set_rerun", analysis_config_set_rerun);
    add_function("analysis_config_get_rerun_start",
                 analysis_config_get_rerun_start);
    add_function("analysis_config_set_rerun_start",
                 analysis_config_set_rerun_start);
    add_function("analysis_config_get_log_path", analysis_config_get_log_path);
    add_function("analysis_config_set_log_path", analysis_config_set_log_path);
    add_function("analysis_config_get_iter_config",
                 analysis_config_get_iter_config, return_ref);
    add_function("analysis_config_get_max_runtime",
                 analysis_config_get_max_runtime);
    add_function("analysis_config_set_max_runtime",
                 analysis_config_set_max_runtime);
    add_function("analysis_config_get_stop_long_running",
                 analysis_config_get_stop_long_running);
    add_function("analysis_config_set_stop_long_running",
                 analysis_config_set_stop_long_running);
    add_function("analysis_config_get_active_module_name",
                 analysis_config_get_active_module_name);
    add_function("analysis_config_get_module", analysis_config_get_module,
                 return_ref);
    add_function("analysis_config_select_module",
                 analysis_config_select_module);
    add_function("analysis_config_has_module", analysis_config_has_module);
    add_function("analysis_config_get_alpha", analysis_config_get_alpha);
    add_function("analysis_config_set_alpha", analysis_config_set_alpha);
    add_function("analysis_config_get_std_cutoff",
                 analysis_config_get_std_cutoff);
    add_function("analysis_config_set_std_cutoff",
                 analysis_config_set_std_cutoff);
    add_function("analysis_config_set_global_std_scaling",
                 analysis_config_set_global_std_scaling);
    add_function("analysis_config_get_global_std_scaling",
                 analysis_config_get_global_std_scaling);
    add_function("analysis_config_get_min_realisations",
                 analysis_config_get_min_realisations);
    // AnalysisIterConfig
    add_function("analysis_iter_config_alloc", analysis_iter_config_alloc,
                 unbound);
    add_function("analysis_iter_config_alloc_full",
                 analysis_iter_config_alloc_full, unbound);
    add_function("analysis_iter_config_free", analysis_iter_config_free);
    add_function("analysis_iter_config_set_num_iterations",
                 analysis_iter_config_set_num_iterations);
    add_function("analysis_iter_config_get_num_iterations",
                 analysis_iter_config_get_num_iterations);
    add_function("analysis_iter_config_get_num_retries_per_iteration",
                 analysis_iter_config_get_num_retries_per_iteration);
    add_function("analysis_iter_config_num_iterations_set",
                 analysis_iter_config_num_iterations_set);
    add_function("analysis_iter_config_set_case_fmt",
                 analysis_iter_config_set_case_fmt);
    add_function("analysis_iter_config_get_case_fmt",
                 analysis_iter_config_get_case_fmt);
    add_function("analysis_iter_config_case_fmt_set",
                 analysis_iter_config_case_fmt_set);
    // EclConfig
    add_function("ecl_config_alloc", ecl_config_alloc, unbound);
    add_function("ecl_config_alloc_full", ecl_config_alloc_full, unbound);
    add_function("ecl_config_free", ecl_config_free);
    add_function("ecl_config_get_data_file", ecl_config_get_data_file);
    add_function("ecl_config_set_data_file", ecl_config_set_data_file);
    add_function("ecl_config_validate_data_file",
                 ecl_config_validate_data_file);
    add_function("ecl_config_get_gridfile", ecl_config_get_gridfile);
    add_function("ecl_config_set_grid", ecl_config_set_grid);
    add_function("ecl_config_validate_grid", ecl_config_validate_grid);
    add_function("ecl_config_get_grid", ecl_config_get_grid, return_ref);
    add_function("ecl_config_get_refcase_name", ecl_config_get_refcase_name);
    add_function("ecl_config_get_refcase", ecl_config_get_refcase, return_ref);
    add_function("ecl_config_load_refcase", ecl_config_load_refcase);
    add_function("ecl_config_validate_refcase", ecl_config_validate_refcase);
    add_function("ecl_config_has_refcase", ecl_config_has_refcase);
    add_function("ecl_config_get_depth_unit", ecl_config_get_depth_unit);
    add_function("ecl_config_get_pressure_unit", ecl_config_get_pressure_unit);
    add_function("ecl_config_active", ecl_config_active);
    add_function("ecl_config_get_last_history_restart",
                 ecl_config_get_last_history_restart);
    add_function("ecl_config_get_end_date", ecl_config_get_end_date);
    add_function("ecl_config_get_num_cpu", ecl_config_get_num_cpu);
    // EnkfFs
    add_function("enkf_fs_mount", enkf_fs_mount, unbound);
    add_function("enkf_fs_sync", enkf_fs_sync);
    add_function("enkf_fs_exists", enkf_fs_exists, unbound);
    add_function("enkf_fs_disk_version", enkf_fs_disk_version, unbound);
    add_function("enkf_fs_update_disk_version", enkf_fs_update_disk_version,
                 unbound);
    add_function("enkf_fs_decref", enkf_fs_decref);
    add_function("enkf_fs_incref", enkf_fs_incref);
    add_function("enkf_fs_get_refcount", enkf_fs_get_refcount);
    add_function("enkf_fs_get_case_name", enkf_fs_get_case_name);
    add_function("enkf_fs_is_read_only", enkf_fs_is_read_only);
    add_function("enkf_fs_is_running", enkf_fs_is_running);
    add_function("enkf_fs_fsync", enkf_fs_fsync);
    add_function("enkf_fs_create_fs", enkf_fs_create_fs, unbound);
    add_function("enkf_fs_get_time_map", enkf_fs_get_time_map, return_ref);
    add_function("enkf_fs_get_state_map", enkf_fs_get_state_map, return_ref);
    add_function("enkf_fs_get_summary_key_set", enkf_fs_get_summary_key_set,
                 return_ref);
    // EnkfFsManager
    add_function("enkf_main_get_fs_ref", enkf_main_get_fs_ref);
    add_function("enkf_main_set_fs", enkf_main_set_fs);
    add_function("enkf_main_alloc_caselist", enkf_main_alloc_caselist);
    add_function("enkf_main_case_is_initialized",
                 enkf_main_case_is_initialized);
    add_function("enkf_main_init_case_from_existing",
                 enkf_main_init_case_from_existing);
    add_function("enkf_main_init_current_case_from_existing",
                 enkf_main_init_current_case_from_existing);
    add_function("enkf_main_alloc_readonly_state_map",
                 enkf_main_alloc_readonly_state_map);
    // EnKFMain
    add_function("enkf_main_alloc", enkf_main_alloc, unbound);
    add_function("enkf_main_free", enkf_main_free);
    add_function("enkf_main_get_queue_config", enkf_main_get_queue_config,
                 return_ref);
    add_function("enkf_main_get_ensemble_size", enkf_main_get_ensemble_size);
    add_function("enkf_main_get_ensemble_config", enkf_main_get_ensemble_config,
                 return_ref);
    add_function("enkf_main_get_model_config", enkf_main_get_model_config,
                 return_ref);
    add_function("enkf_main_get_local_config", enkf_main_get_local_config,
                 return_ref);
    add_function("enkf_main_get_analysis_config", enkf_main_get_analysis_config,
                 return_ref);
    add_function("enkf_main_get_site_config", enkf_main_get_site_config,
                 return_ref);
    add_function("enkf_main_get_ecl_config", enkf_main_get_ecl_config,
                 return_ref);
    add_function("enkf_main_get_schedule_prediction_file",
                 enkf_main_get_schedule_prediction_file);
    add_function("enkf_main_get_data_kw", enkf_main_get_data_kw, return_ref);
    add_function("enkf_main_clear_data_kw", enkf_main_clear_data_kw);
    add_function("enkf_main_add_data_kw", enkf_main_add_data_kw);
    add_function("enkf_main_resize_ensemble", enkf_main_resize_ensemble);
    add_function("enkf_main_get_obs", enkf_main_get_obs, return_ref);
    add_function("enkf_main_load_obs", enkf_main_load_obs);
    add_function("enkf_main_get_templates", enkf_main_get_templates,
                 return_ref);
    add_function("enkf_main_get_site_config_file",
                 enkf_main_get_site_config_file);
    add_function("enkf_main_get_history_length", enkf_main_get_history_length);
    add_function("enkf_main_get_observation_count",
                 enkf_main_get_observation_count);
    add_function("enkf_main_have_obs", enkf_main_have_obs);
    add_function("enkf_main_iget_state", enkf_main_iget_state, return_ref);
    add_function("enkf_main_get_workflow_list", enkf_main_get_workflow_list,
                 return_ref);
    add_function("enkf_main_get_hook_manager", enkf_main_get_hook_manager,
                 return_ref);
    add_function("enkf_main_get_user_config_file",
                 enkf_main_get_user_config_file);
    add_function("enkf_main_get_mount_root", enkf_main_get_mount_root);
    add_function("enkf_main_export_field_with_fs",
                 enkf_main_export_field_with_fs);
    add_function("enkf_main_load_from_forward_model_from_gui",
                 enkf_main_load_from_forward_model_from_gui);
    add_function("enkf_main_load_from_run_context_from_gui",
                 enkf_main_load_from_run_context_from_gui);
    add_function("enkf_main_create_run_path", enkf_main_create_run_path);
    add_function("enkf_main_alloc_ert_run_context_ENSEMBLE_EXPERIMENT",
                 enkf_main_alloc_ert_run_context_ENSEMBLE_EXPERIMENT);
    add_function("enkf_main_get_runpath_list", enkf_main_get_runpath_list,
                 return_ref);
    add_function("enkf_main_alloc_runpath_list", enkf_main_alloc_runpath_list);
    add_function("enkf_main_add_node", enkf_main_add_node);
    add_function("enkf_main_get_res_config", enkf_main_get_res_config,
                 return_ref);
    add_function("enkf_main_init_run", enkf_main_init_run);
    add_function("enkf_main_get_shared_rng", enkf_main_get_shared_rng,
                 return_ref);
    // EnkfObs
    add_function("enkf_obs_alloc", enkf_obs_alloc, unbound);
    add_function("enkf_obs_free", enkf_obs_free);
    add_function("enkf_obs_get_size", enkf_obs_get_size);
    add_function("enkf_obs_is_valid", enkf_obs_is_valid);
    add_function("enkf_obs_load", enkf_obs_load);
    add_function("enkf_obs_clear", enkf_obs_clear);
    add_function("enkf_obs_alloc_typed_keylist", enkf_obs_alloc_typed_keylist);
    add_function("enkf_obs_alloc_matching_keylist",
                 enkf_obs_alloc_matching_keylist);
    add_function("enkf_obs_has_key", enkf_obs_has_key);
    add_function("enkf_obs_get_type", enkf_obs_get_type);
    add_function("enkf_obs_get_vector", enkf_obs_get_vector, return_ref);
    add_function("enkf_obs_iget_vector", enkf_obs_iget_vector, return_ref);
    add_function("enkf_obs_iget_obs_time", enkf_obs_iget_obs_time);
    add_function("enkf_obs_add_obs_vector", enkf_obs_add_obs_vector);
    add_function("enkf_obs_get_obs_and_measure_data",
                 enkf_obs_get_obs_and_measure_data);
    add_function("enkf_obs_alloc_all_active_local_obs",
                 enkf_obs_alloc_all_active_local_obs);
    add_function("enkf_obs_local_scale_std", enkf_obs_local_scale_std);
    // EnKFState
    add_function("enkf_state_free", enkf_state_free);
    add_function("enkf_state_get_ensemble_config",
                 enkf_state_get_ensemble_config, return_ref);
    add_function("enkf_state_initialize", enkf_state_initialize);
    add_function("enkf_state_complete_forward_modelOK",
                 enkf_state_complete_forward_modelOK, unbound);
    add_function("enkf_state_complete_forward_model_EXIT_handler__",
                 enkf_state_complete_forward_model_EXIT_handler__, unbound);
    // EnsembleConfig
    add_function("ensemble_config_alloc", ensemble_config_alloc, unbound);
    add_function("ensemble_config_alloc_full", ensemble_config_alloc_full,
                 unbound);
    add_function("ensemble_config_free", ensemble_config_free);
    add_function("ensemble_config_has_key", ensemble_config_has_key);
    add_function("ensemble_config_get_size", ensemble_config_get_size);
    add_function("ensemble_config_get_node", ensemble_config_get_node,
                 return_ref);
    add_function("ensemble_config_alloc_keylist",
                 ensemble_config_alloc_keylist);
    add_function("ensemble_config_add_summary", ensemble_config_add_summary,
                 return_ref);
    add_function("ensemble_config_add_gen_kw", ensemble_config_add_gen_kw,
                 return_ref);
    add_function("ensemble_config_add_field", ensemble_config_add_field,
                 return_ref);
    add_function("ensemble_config_alloc_keylist_from_impl_type",
                 ensemble_config_alloc_keylist_from_impl_type);
    add_function("ensemble_config_add_node", ensemble_config_add_node);
    add_function("ensemble_config_get_summary_key_matcher",
                 ensemble_config_get_summary_key_matcher, return_ref);
    add_function("ensemble_config_get_trans_table",
                 ensemble_config_get_trans_table);
    add_function("ensemble_config_init_SUMMARY_full",
                 ensemble_config_init_SUMMARY_full);
    // ErtRunContext
    add_function("ert_run_context_alloc", ert_run_context_alloc, unbound);
    add_function("ert_run_context_alloc_ENSEMBLE_EXPERIMENT",
                 ert_run_context_alloc_ENSEMBLE_EXPERIMENT, unbound);
    add_function("ert_run_context_alloc_SMOOTHER_RUN",
                 ert_run_context_alloc_SMOOTHER_RUN, unbound);
    add_function("ert_run_context_alloc_SMOOTHER_UPDATE",
                 ert_run_context_alloc_SMOOTHER_UPDATE, unbound);
    add_function("ert_run_context_alloc_CASE_INIT",
                 ert_run_context_alloc_CASE_INIT, unbound);
    add_function("ert_run_context_alloc_runpath_list",
                 ert_run_context_alloc_runpath_list, unbound);
    add_function("ert_run_context_alloc_runpath", ert_run_context_alloc_runpath,
                 unbound);
    add_function("ert_run_context_get_size", ert_run_context_get_size);
    add_function("ert_run_context_free", ert_run_context_free);
    add_function("ert_run_context_iactive", ert_run_context_iactive);
    add_function("ert_run_context_iget_arg", ert_run_context_iget_arg,
                 return_ref);
    add_function("ert_run_context_get_id", ert_run_context_get_id);
    add_function("ert_run_context_alloc_iactive",
                 ert_run_context_alloc_iactive);
    add_function("ert_run_context_get_iter", ert_run_context_get_iter);
    add_function("ert_run_context_get_update_target_fs",
                 ert_run_context_get_update_target_fs, return_ref);
    add_function("ert_run_context_get_sim_fs", ert_run_context_get_sim_fs,
                 return_ref);
    add_function("ert_run_context_get_init_mode",
                 ert_run_context_get_init_mode);
    add_function("ert_run_context_get_step1", ert_run_context_get_step1);
    add_function("ert_run_context_deactivate_realization",
                 ert_run_context_deactivate_realization);
    // ErtTemplates
    add_function("ert_templates_alloc", ert_templates_alloc, unbound);
    add_function("ert_templates_alloc_default", ert_templates_alloc_default,
                 unbound);
    add_function("ert_templates_free", ert_templates_free);
    add_function("ert_templates_alloc_list", ert_templates_alloc_list,
                 return_ref);
    add_function("ert_templates_get_template", ert_templates_get_template,
                 return_ref);
    add_function("ert_templates_clear", ert_templates_clear);
    add_function("ert_templates_add_template", ert_templates_add_template,
                 return_ref);
    // ErtWorkflowList
    add_function("ert_workflow_list_alloc", ert_workflow_list_alloc, unbound);
    add_function("ert_workflow_list_alloc_full", ert_workflow_list_alloc_full,
                 unbound);
    add_function("ert_workflow_list_free", ert_workflow_list_free);
    add_function("ert_workflow_list_alloc_namelist",
                 ert_workflow_list_alloc_namelist);
    add_function("ert_workflow_list_has_workflow",
                 ert_workflow_list_has_workflow);
    add_function("ert_workflow_list_get_workflow",
                 ert_workflow_list_get_workflow, return_ref);
    add_function("ert_workflow_list_add_workflow",
                 ert_workflow_list_add_workflow, return_ref);
    add_function("ert_workflow_list_get_context", ert_workflow_list_get_context,
                 return_ref);
    add_function("ert_workflow_list_add_job", ert_workflow_list_add_job);
    add_function("ert_workflow_list_has_job", ert_workflow_list_has_job);
    add_function("ert_workflow_list_get_job", ert_workflow_list_get_job,
                 return_ref);
    add_function("ert_workflow_list_get_job_names",
                 ert_workflow_list_get_job_names);
    // ESUpdate
    add_function("enkf_main_smoother_update", enkf_main_smoother_update);
    // ForwardLoadContext
    add_function("forward_load_context_alloc", forward_load_context_alloc,
                 unbound);
    add_function("forward_load_context_select_step",
                 forward_load_context_select_step);
    add_function("forward_load_context_get_load_step",
                 forward_load_context_get_load_step);
    add_function("forward_load_context_free", forward_load_context_free);
    // HookManager
    add_function("hook_manager_alloc", hook_manager_alloc, unbound);
    add_function("hook_manager_alloc_full", hook_manager_alloc_full, unbound);
    add_function("hook_manager_free", hook_manager_free);
    add_function("hook_manager_get_runpath_list_file",
                 hook_manager_get_runpath_list_file);
    add_function("hook_manager_get_runpath_list", hook_manager_get_runpath_list,
                 return_ref);
    add_function("hook_manager_iget_hook_workflow",
                 hook_manager_iget_hook_workflow, return_ref);
    add_function("hook_manager_get_size", hook_manager_get_size);
    // HookWorkflow
    add_function("hook_workflow_get_workflow", hook_workflow_get_workflow,
                 return_ref);
    add_function("hook_workflow_get_run_mode", hook_workflow_get_run_mode);
    // LocalConfig
    add_function("local_config_free", local_config_free);
    add_function("local_config_clear", local_config_clear);
    add_function("local_config_clear_active", local_config_clear_active);
    add_function("local_config_alloc_ministep", local_config_alloc_ministep,
                 return_ref);
    add_function("local_updatestep_add_ministep", local_updatestep_add_ministep,
                 unbound);
    add_function("local_config_alloc_obsdata", local_config_alloc_obsdata);
    add_function("local_config_has_obsdata", local_config_has_obsdata);
    add_function("local_config_get_updatestep", local_config_get_updatestep,
                 return_ref);
    add_function("local_config_get_ministep", local_config_get_ministep,
                 return_ref);
    add_function("local_config_get_obsdata", local_config_get_obsdata,
                 return_ref);
    add_function("local_config_alloc_obsdata_copy",
                 local_config_alloc_obsdata_copy, return_ref);
    // LocalMinistep
    add_function("local_ministep_add_obsdata_node",
                 local_ministep_add_obsdata_node);
    add_function("local_ministep_get_obsdata", local_ministep_get_obsdata,
                 return_ref);
    add_function("local_ministep_get_obs_data", local_ministep_get_obs_data,
                 return_ref);
    add_function("local_ministep_free", local_ministep_free);
    add_function("local_ministep_add_obsdata", local_ministep_add_obsdata);
    add_function("local_ministep_get_name", local_ministep_get_name);
    add_function("local_ministep_data_is_active", local_ministep_data_is_active);
    add_function("local_ministep_activate_data", local_ministep_activate_data);
    add_function("local_ministep_get_or_create_row_scaling", local_ministep_get_or_create_row_scaling, return_ref);
    // LocalObsdata
    add_function("local_obsdata_alloc", local_obsdata_alloc, unbound);
    add_function("local_obsdata_free", local_obsdata_free);
    add_function("local_obsdata_get_size", local_obsdata_get_size);
    add_function("local_obsdata_has_node", local_obsdata_has_node);
    add_function("local_obsdata_add_node", local_obsdata_add_node);
    add_function("local_obsdata_del_node", local_obsdata_del_node);
    add_function("local_obsdata_get_name", local_obsdata_get_name);
    add_function("local_obsdata_iget", local_obsdata_iget, return_ref);
    add_function("local_obsdata_get", local_obsdata_get, return_ref);
    add_function("local_obsdata_get_copy_node_active_list",
                 local_obsdata_get_copy_node_active_list, return_ref);
    add_function("local_obsdata_get_node_active_list",
                 local_obsdata_get_node_active_list, return_ref);
    // LocalObsdataNode
    add_function("local_obsdata_node_alloc", local_obsdata_node_alloc, unbound);
    add_function("local_obsdata_node_free", local_obsdata_node_free);
    add_function("local_obsdata_node_get_key", local_obsdata_node_get_key);
    add_function("local_obsdata_node_add_tstep", local_obsdata_node_add_tstep);
    add_function("local_obsdata_node_tstep_active",
                 local_obsdata_node_tstep_active);
    add_function("local_obsdata_node_all_timestep_active",
                 local_obsdata_node_all_timestep_active);
    add_function("local_obsdata_node_set_all_timestep_active",
                 local_obsdata_node_set_all_timestep_active);
    add_function("local_obsdata_node_get_active_list",
                 local_obsdata_node_get_active_list, return_ref);
    // LocalUpdateStep
    add_function("local_updatestep_get_num_ministep",
                 local_updatestep_get_num_ministep);
    add_function("local_updatestep_iget_ministep",
                 local_updatestep_iget_ministep, return_ref);
    add_function("local_updatestep_free", local_updatestep_free);
    add_function("local_updatestep_get_name", local_updatestep_get_name);
    // LogConfig
    add_function("log_config_alloc", log_config_alloc, unbound);
    add_function("log_config_alloc_load", log_config_alloc_load, unbound);
    add_function("log_config_alloc_full", log_config_alloc_full, unbound);
    add_function("log_config_free", log_config_free);
    add_function("log_config_get_log_file", log_config_get_log_file);
    add_function("log_config_get_log_level", log_config_get_log_level);
    // MeasBlock
    add_function("meas_block_alloc", meas_block_alloc, unbound);
    add_function("meas_block_free", meas_block_free);
    add_function("meas_block_get_active_ens_size",
                 meas_block_get_active_ens_size);
    add_function("meas_block_get_total_ens_size",
                 meas_block_get_total_ens_size);
    add_function("meas_block_get_total_obs_size",
                 meas_block_get_total_obs_size);
    add_function("meas_block_iget", meas_block_iget);
    add_function("meas_block_iset", meas_block_iset);
    add_function("meas_block_iget_ens_mean", meas_block_iget_ens_mean);
    add_function("meas_block_iget_ens_std", meas_block_iget_ens_std);
    add_function("meas_block_iens_active", meas_block_iens_active);
    // MeasData
    add_function("meas_data_alloc", meas_data_alloc, unbound);
    add_function("meas_data_free", meas_data_free);
    add_function("meas_data_get_active_obs_size",
                 meas_data_get_active_obs_size);
    add_function("meas_data_get_active_ens_size",
                 meas_data_get_active_ens_size);
    add_function("meas_data_get_total_ens_size", meas_data_get_total_ens_size);
    add_function("meas_data_get_num_blocks", meas_data_get_num_blocks);
    add_function("meas_data_has_block", meas_data_has_block);
    add_function("meas_data_get_block", meas_data_get_block, return_ref);
    add_function("meas_data_allocS", meas_data_allocS);
    add_function("meas_data_add_block", meas_data_add_block, return_ref);
    add_function("meas_data_iget_block", meas_data_iget_block, return_ref);
    add_function("enkf_analysis_deactivate_std_zero",
                 enkf_analysis_deactivate_std_zero, unbound);
    // ModelConfig
    add_function("model_config_alloc", model_config_alloc, unbound);
    add_function("model_config_alloc_full", model_config_alloc_full, unbound);
    add_function("model_config_free", model_config_free);
    add_function("model_config_get_forward_model",
                 model_config_get_forward_model, return_ref);
    add_function("model_config_get_max_internal_submit",
                 model_config_get_max_internal_submit);
    add_function("model_config_get_runpath_as_char",
                 model_config_get_runpath_as_char);
    add_function("model_config_select_runpath", model_config_select_runpath);
    add_function("model_config_set_runpath", model_config_set_runpath);
    add_function("model_config_get_enspath", model_config_get_enspath);
    add_function("model_config_get_history_source",
                 model_config_get_history_source);
    add_function("model_config_select_history", model_config_select_history);
    add_function("model_config_has_history", model_config_has_history);
    add_function("model_config_get_gen_kw_export_name",
                 model_config_get_gen_kw_export_name);
    add_function("model_config_runpath_requires_iter",
                 model_config_runpath_requires_iter);
    add_function("model_config_get_jobname_fmt", model_config_get_jobname_fmt);
    add_function("model_config_get_runpath_fmt", model_config_get_runpath_fmt,
                 return_ref);
    add_function("model_config_get_num_realizations",
                 model_config_get_num_realizations);
    add_function("model_config_get_obs_config_file",
                 model_config_get_obs_config_file);
    add_function("model_config_get_data_root", model_config_get_data_root);
    add_function("model_config_get_external_time_map",
                 model_config_get_external_time_map);
    // ObsBlock
    add_function("obs_block_alloc", obs_block_alloc, unbound);
    add_function("obs_block_free", obs_block_free);
    add_function("obs_block_get_size", obs_block_get_size);
    add_function("obs_block_get_active_size", obs_block_get_active_size);
    add_function("obs_block_iset", obs_block_iset);
    add_function("obs_block_iget_value", obs_block_iget_value);
    add_function("obs_block_iget_std", obs_block_iget_std);
    add_function("obs_block_get_key", obs_block_get_key);
    add_function("obs_block_iget_is_active", obs_block_iget_is_active);
    // ObsData
    add_function("obs_data_alloc", obs_data_alloc, unbound);
    add_function("obs_data_free", obs_data_free);
    add_function("obs_data_get_total_size", obs_data_get_total_size);
    add_function("obs_data_scale", obs_data_scale);
    add_function("obs_data_scale_matrix", obs_data_scale_matrix);
    add_function("obs_data_scale_Rmatrix", obs_data_scale_Rmatrix);
    add_function("obs_data_iget_value", obs_data_iget_value);
    add_function("obs_data_iget_std", obs_data_iget_std);
    add_function("obs_data_add_block", obs_data_add_block, return_ref);
    add_function("obs_data_allocdObs", obs_data_allocdObs);
    add_function("obs_data_allocR", obs_data_allocR);
    add_function("obs_data_allocD", obs_data_allocD);
    add_function("obs_data_allocE", obs_data_allocE);
    add_function("obs_data_iget_block", obs_data_iget_block, return_ref);
    add_function("obs_data_get_num_blocks", obs_data_get_num_blocks);
    // QueueConfig
    add_function("queue_config_free", queue_config_free);
    add_function("queue_config_alloc_load", queue_config_alloc_load, unbound);
    add_function("queue_config_alloc_full", queue_config_alloc_full, unbound);
    add_function("queue_config_alloc", queue_config_alloc, unbound);
    add_function("queue_config_alloc_local_copy",
                 queue_config_alloc_local_copy);
    add_function("queue_config_has_job_script", queue_config_has_job_script);
    add_function("queue_config_get_job_script", queue_config_get_job_script);
    add_function("queue_config_get_max_submit", queue_config_get_max_submit);
    add_function("queue_config_get_queue_system",
                 queue_config_get_queue_system);
    add_function("queue_config_get_queue_driver", queue_config_get_queue_driver,
                 return_ref);
    add_function("queue_config_get_num_cpu", queue_config_get_num_cpu);
    add_function("queue_config_lsf_queue_name", queue_config_lsf_queue_name,
                 unbound);
    add_function("queue_config_lsf_server", queue_config_lsf_server, unbound);
    add_function("queue_config_lsf_resource", queue_config_lsf_resource,
                 unbound);
    add_function("queue_config_lsf_driver_name", queue_config_lsf_driver_name,
                 unbound);
    // ResConfig
    add_function("res_config_free", res_config_free);
    add_function("res_config_alloc_full", res_config_alloc_full, unbound);
    add_function("res_config_alloc_user_content", res_config_alloc_user_content,
                 unbound | return_ref);
    add_function("res_config_get_user_config_file",
                 res_config_get_user_config_file);
    add_function("res_config_get_config_directory",
                 res_config_get_config_directory);
    add_function("res_config_get_site_config", res_config_get_site_config,
                 return_ref);
    add_function("res_config_get_analysis_config",
                 res_config_get_analysis_config, return_ref);
    add_function("res_config_get_subst_config", res_config_get_subst_config,
                 return_ref);
    add_function("res_config_get_model_config", res_config_get_model_config,
                 return_ref);
    add_function("res_config_get_ecl_config", res_config_get_ecl_config,
                 return_ref);
    add_function("res_config_get_ensemble_config",
                 res_config_get_ensemble_config, return_ref);
    add_function("res_config_get_hook_manager", res_config_get_hook_manager,
                 return_ref);
    add_function("res_config_get_workflow_list", res_config_get_workflow_list,
                 return_ref);
    add_function("res_config_get_rng_config", res_config_get_rng_config,
                 return_ref);
    add_function("res_config_get_templates", res_config_get_templates,
                 return_ref);
    add_function("res_config_get_log_config", res_config_get_log_config,
                 return_ref);
    add_function("res_config_get_queue_config", res_config_get_queue_config,
                 return_ref);
    add_function("res_config_init_config_parser", res_config_init_config_parser,
                 unbound);
    // RNGConfig
    add_function("rng_config_alloc", rng_config_alloc, unbound);
    add_function("rng_config_alloc_full", rng_config_alloc_full, unbound);
    add_function("rng_config_free", rng_config_free);
    add_function("rng_config_get_type", rng_config_get_type);
    add_function("rng_config_get_random_seed", rng_config_get_random_seed);
    // RowScaling
    add_function("row_scaling_alloc", row_scaling_alloc, unbound);
    add_function("row_scaling_free", row_scaling_free);
    add_function("row_scaling_get_size", row_scaling_get_size);
    add_function("row_scaling_iset", row_scaling_iset);
    add_function("row_scaling_iget", row_scaling_iget);
    add_function("row_scaling_clamp", row_scaling_clamp);
    // add_function("row_scaling_assign_double", row_scaling_assign_double);
    // add_function("row_scaling_assign_float", row_scaling_assign_float);
    // RunArg
    add_function("run_arg_alloc_ENSEMBLE_EXPERIMENT",
                 run_arg_alloc_ENSEMBLE_EXPERIMENT, unbound);
    add_function("run_arg_free", run_arg_free);
    add_function("run_arg_get_queue_index_safe", run_arg_get_queue_index_safe);
    add_function("run_arg_set_queue_index", run_arg_set_queue_index);
    add_function("run_arg_is_submitted", run_arg_is_submitted);
    add_function("run_arg_get_run_id", run_arg_get_run_id);
    add_function("run_arg_get_geo_id", run_arg_get_geo_id);
    add_function("run_arg_set_geo_id", run_arg_set_geo_id);
    add_function("run_arg_get_runpath", run_arg_get_runpath);
    add_function("run_arg_get_iter", run_arg_get_iter);
    add_function("run_arg_get_iens", run_arg_get_iens);
    add_function("run_arg_get_run_status", run_arg_get_run_status);
    add_function("run_arg_get_job_name", run_arg_get_job_name);
    // RunpathList
    add_function("runpath_list_alloc", runpath_list_alloc, unbound);
    add_function("runpath_list_free", runpath_list_free);
    add_function("runpath_list_add", runpath_list_add);
    add_function("runpath_list_clear", runpath_list_clear);
    add_function("runpath_list_size", runpath_list_size);
    add_function("runpath_list_iget_iens", runpath_list_iget_iens);
    add_function("runpath_list_iget_iter", runpath_list_iget_iter);
    add_function("runpath_list_iget_runpath", runpath_list_iget_runpath);
    add_function("runpath_list_iget_basename", runpath_list_iget_basename);
    add_function("runpath_list_fprintf", runpath_list_fprintf);
    add_function("runpath_list_load", runpath_list_load);
    add_function("runpath_list_get_export_file", runpath_list_get_export_file);
    add_function("runpath_list_set_export_file", runpath_list_set_export_file);
    // SiteConfig
    add_function("site_config_alloc", site_config_alloc, unbound);
    add_function("site_config_alloc_full", site_config_alloc_full, unbound);
    add_function("site_config_alloc_load_user_config",
                 site_config_alloc_load_user_config, unbound);
    add_function("site_config_free", site_config_free);
    add_function("site_config_get_installed_jobs",
                 site_config_get_installed_jobs, return_ref);
    add_function("site_config_get_license_root_path",
                 site_config_get_license_root_path);
    add_function("site_config_set_license_root_path",
                 site_config_set_license_root_path);
    add_function("site_config_get_location", site_config_get_location, unbound);
    add_function("site_config_get_config_file", site_config_get_config_file);
    add_function("site_config_get_umask", site_config_get_umask);
    // StateMap
    add_function("state_map_alloc", state_map_alloc, unbound);
    add_function("state_map_fread", state_map_fread);
    add_function("state_map_fwrite", state_map_fwrite);
    add_function("state_map_equal", state_map_equal);
    add_function("state_map_free", state_map_free);
    add_function("state_map_get_size", state_map_get_size);
    add_function("state_map_iget", state_map_iget);
    add_function("state_map_iset", state_map_iset);
    add_function("state_map_select_matching", state_map_select_matching);
    add_function("state_map_is_readonly", state_map_is_readonly);
    add_function("state_map_legal_transition", state_map_legal_transition,
                 unbound);
    // SubstConfig
    add_function("subst_config_alloc", subst_config_alloc, unbound);
    add_function("subst_config_alloc_full", subst_config_alloc_full, unbound);
    add_function("subst_config_free", subst_config_free);
    add_function("subst_config_get_subst_list", subst_config_get_subst_list,
                 return_ref);
    add_function("ecl_util_get_num_cpu", ecl_util_get_num_cpu, unbound);
    // SummaryKeyMatcher
    add_function("summary_key_matcher_alloc", summary_key_matcher_alloc,
                 unbound);
    add_function("summary_key_matcher_free", summary_key_matcher_free);
    add_function("summary_key_matcher_get_size", summary_key_matcher_get_size);
    add_function("summary_key_matcher_add_summary_key",
                 summary_key_matcher_add_summary_key);
    add_function("summary_key_matcher_match_summary_key",
                 summary_key_matcher_match_summary_key);
    add_function("summary_key_matcher_get_keys", summary_key_matcher_get_keys);
    add_function("summary_key_matcher_summary_key_is_required",
                 summary_key_matcher_summary_key_is_required);
    // SummaryKeySet
    add_function("summary_key_set_alloc", summary_key_set_alloc, unbound);
    add_function("summary_key_set_alloc_from_file",
                 summary_key_set_alloc_from_file, unbound);
    add_function("summary_key_set_free", summary_key_set_free);
    add_function("summary_key_set_get_size", summary_key_set_get_size);
    add_function("summary_key_set_add_summary_key",
                 summary_key_set_add_summary_key);
    add_function("summary_key_set_has_summary_key",
                 summary_key_set_has_summary_key);
    add_function("summary_key_set_alloc_keys", summary_key_set_alloc_keys);
    add_function("summary_key_set_is_read_only", summary_key_set_is_read_only);
    add_function("summary_key_set_fwrite", summary_key_set_fwrite);
    // ActiveList
    add_function("active_list_alloc", active_list_alloc, unbound);
    add_function("active_list_free", active_list_free);
    add_function("active_list_add_index", active_list_add_index);
    add_function("active_list_get_active_size", active_list_get_active_size);
    add_function("active_list_get_mode", active_list_get_mode);
    // EnkfConfigNode
    add_function("enkf_config_node_alloc", enkf_config_node_alloc, unbound);
    add_function("enkf_config_node_alloc_GEN_DATA_everest",
                 enkf_config_node_alloc_GEN_DATA_everest, unbound);
    add_function("enkf_config_node_alloc_summary",
                 enkf_config_node_alloc_summary, unbound);
    add_function("enkf_config_node_alloc_field", enkf_config_node_alloc_field,
                 unbound);
    add_function("enkf_config_node_get_ref", enkf_config_node_get_ref);
    add_function("enkf_config_node_get_impl_type",
                 enkf_config_node_get_impl_type);
    add_function("enkf_config_node_get_enkf_outfile",
                 enkf_config_node_get_enkf_outfile);
    add_function("enkf_config_node_get_min_std_file",
                 enkf_config_node_get_min_std_file);
    add_function("enkf_config_node_get_enkf_infile",
                 enkf_config_node_get_enkf_infile);
    add_function("enkf_config_node_get_init_file_fmt",
                 enkf_config_node_get_init_file_fmt);
    add_function("enkf_config_node_get_var_type",
                 enkf_config_node_get_var_type);
    add_function("enkf_config_node_get_key", enkf_config_node_get_key);
    add_function("enkf_config_node_get_obs_keys", enkf_config_node_get_obs_keys,
                 return_ref);
    add_function("enkf_config_node_free", enkf_config_node_free);
    add_function("enkf_config_node_use_forward_init",
                 enkf_config_node_use_forward_init);
    add_function("enkf_config_node_alloc_GEN_PARAM_full",
                 enkf_config_node_alloc_GEN_PARAM_full, unbound);
    add_function("enkf_config_node_alloc_GEN_DATA_full",
                 enkf_config_node_alloc_GEN_DATA_full, unbound);
    add_function("enkf_config_node_alloc_GEN_KW_full",
                 enkf_config_node_alloc_GEN_KW_full, unbound);
    add_function("enkf_config_node_alloc_SURFACE_full",
                 enkf_config_node_alloc_SURFACE_full, unbound);
    add_function("enkf_config_node_new_container",
                 enkf_config_node_new_container, unbound);
    add_function("enkf_config_node_update_container",
                 enkf_config_node_update_container);
    add_function("enkf_config_node_container_size",
                 enkf_config_node_container_size);
    add_function("enkf_config_node_iget_container_key",
                 enkf_config_node_iget_container_key);
    add_function("enkf_config_node_update_parameter_field",
                 enkf_config_node_update_parameter_field);
    add_function("enkf_config_node_update_general_field",
                 enkf_config_node_update_general_field);
    // ExtParamConfig
    add_function("ext_param_config_alloc", ext_param_config_alloc, unbound);
    add_function("ext_param_config_get_data_size",
                 ext_param_config_get_data_size);
    add_function("ext_param_config_iget_key", ext_param_config_iget_key);
    add_function("ext_param_config_free", ext_param_config_free);
    add_function("ext_param_config_has_key", ext_param_config_has_key);
    add_function("ext_param_config_get_key_index",
                 ext_param_config_get_key_index);
    add_function("ext_param_config_ikey_get_suffix_count",
                 ext_param_config_ikey_get_suffix_count);
    add_function("ext_param_config_ikey_iget_suffix",
                 ext_param_config_ikey_iget_suffix);
    add_function("ext_param_config_ikey_set_suffixes",
                 ext_param_config_ikey_set_suffixes);
    // FieldConfig
    add_function("field_config_alloc_empty", field_config_alloc_empty, unbound);
    add_function("field_config_free", field_config_free);
    add_function("field_config_get_type", field_config_get_type);
    add_function("field_config_get_truncation_mode",
                 field_config_get_truncation_mode);
    add_function("field_config_get_truncation_min",
                 field_config_get_truncation_min);
    add_function("field_config_get_truncation_max",
                 field_config_get_truncation_max);
    add_function("field_config_get_init_transform_name",
                 field_config_get_init_transform_name);
    add_function("field_config_get_output_transform_name",
                 field_config_get_output_transform_name);
    add_function("field_config_ijk_active", field_config_ijk_active);
    add_function("field_config_get_nx", field_config_get_nx);
    add_function("field_config_get_ny", field_config_get_ny);
    add_function("field_config_get_nz", field_config_get_nz);
    add_function("field_config_get_grid", field_config_get_grid, return_ref);
    add_function("field_config_get_data_size_from_grid",
                 field_config_get_data_size_from_grid);
    add_function("field_config_default_export_format",
                 field_config_default_export_format, unbound);
    add_function("field_config_guess_file_type", field_config_guess_file_type,
                 unbound);
    // GenDataConfig
    add_function("gen_data_config_alloc_GEN_DATA_result",
                 gen_data_config_alloc_GEN_DATA_result, unbound);
    add_function("gen_data_config_free", gen_data_config_free);
    add_function("gen_data_config_get_output_format",
                 gen_data_config_get_output_format);
    add_function("gen_data_config_get_input_format",
                 gen_data_config_get_input_format);
    add_function("gen_data_config_get_template_file",
                 gen_data_config_get_template_file);
    add_function("gen_data_config_get_template_key",
                 gen_data_config_get_template_key);
    add_function("gen_data_config_get_initial_size",
                 gen_data_config_get_initial_size);
    add_function("gen_data_config_has_report_step",
                 gen_data_config_has_report_step);
    add_function("gen_data_config_get_data_size__",
                 gen_data_config_get_data_size__);
    add_function("gen_data_config_get_key", gen_data_config_get_key);
    add_function("gen_data_config_get_active_mask",
                 gen_data_config_get_active_mask, return_ref);
    add_function("gen_data_config_num_report_step",
                 gen_data_config_num_report_step);
    add_function("gen_data_config_iget_report_step",
                 gen_data_config_iget_report_step);
    // GenKwConfig
    add_function("gen_kw_config_free", gen_kw_config_free);
    add_function("gen_kw_config_alloc_empty", gen_kw_config_alloc_empty,
                 unbound);
    add_function("gen_kw_config_get_template_file",
                 gen_kw_config_get_template_file);
    add_function("gen_kw_config_set_template_file",
                 gen_kw_config_set_template_file);
    add_function("gen_kw_config_get_parameter_file",
                 gen_kw_config_get_parameter_file);
    add_function("gen_kw_config_set_parameter_file",
                 gen_kw_config_set_parameter_file);
    add_function("gen_kw_config_alloc_name_list",
                 gen_kw_config_alloc_name_list);
    add_function("gen_kw_config_should_use_log_scale",
                 gen_kw_config_should_use_log_scale);
    add_function("gen_kw_config_get_key", gen_kw_config_get_key);
    add_function("gen_kw_config_get_tag_fmt", gen_kw_config_get_tag_fmt);
    add_function("gen_kw_config_get_data_size", gen_kw_config_get_data_size);
    add_function("gen_kw_config_iget_name", gen_kw_config_iget_name);
    add_function("gen_kw_config_iget_function_type",
                 gen_kw_config_iget_function_type);
    add_function("gen_kw_config_iget_function_parameter_names",
                 gen_kw_config_iget_function_parameter_names, return_ref);
    add_function("gen_kw_config_iget_function_parameter_values",
                 gen_kw_config_iget_function_parameter_values, return_ref);
    // SummaryConfig
    add_function("summary_config_alloc", summary_config_alloc, unbound);
    add_function("summary_config_free", summary_config_free);
    add_function("summary_config_get_var", summary_config_get_var);
    // Field
    add_function("field_free", field_free);
    add_function("field_get_size", field_get_size);
    add_function("field_ijk_get_double", field_ijk_get_double);
    add_function("field_iget_double", field_iget_double);
    add_function("field_export", field_export);
    // EnkfNode
    add_function("enkf_node_alloc", enkf_node_alloc, unbound);
    add_function("enkf_node_alloc_private_container",
                 enkf_node_alloc_private_container, unbound);
    add_function("enkf_node_free", enkf_node_free);
    add_function("enkf_node_get_key", enkf_node_get_key);
    add_function("enkf_node_value_ptr", enkf_node_value_ptr);
    add_function("enkf_node_try_load", enkf_node_try_load);
    add_function("enkf_node_store", enkf_node_store);
    add_function("enkf_node_get_impl_type", enkf_node_get_impl_type);
    add_function("enkf_node_ecl_write", enkf_node_ecl_write);
    // ExtParam
    add_function("ext_param_alloc", ext_param_alloc, unbound);
    add_function("ext_param_free", ext_param_free);
    add_function("ext_param_iset", ext_param_iset);
    add_function("ext_param_key_set", ext_param_key_set);
    add_function("ext_param_key_suffix_set", ext_param_key_suffix_set);
    add_function("ext_param_iget", ext_param_iget);
    add_function("ext_param_key_get", ext_param_key_get);
    add_function("ext_param_key_suffix_get", ext_param_key_suffix_get);
    add_function("ext_param_json_export", ext_param_json_export);
    add_function("ext_param_get_config", ext_param_get_config);
    // GenData
    add_function("gen_data_alloc", gen_data_alloc, unbound);
    add_function("gen_data_free", gen_data_free);
    add_function("gen_data_get_size", gen_data_get_size);
    add_function("gen_data_iget_double", gen_data_iget_double);
    add_function("gen_data_export", gen_data_export);
    add_function("gen_data_export_data", gen_data_export_data);
    // GenKw
    add_function("gen_kw_alloc", gen_kw_alloc, unbound);
    add_function("gen_kw_free", gen_kw_free);
    add_function("gen_kw_write_export_file", gen_kw_write_export_file);
    add_function("gen_kw_ecl_write_template", gen_kw_ecl_write_template);
    add_function("gen_kw_data_iget", gen_kw_data_iget);
    add_function("gen_kw_data_iset", gen_kw_data_iset);
    add_function("gen_kw_data_set_vector", gen_kw_data_set_vector);
    add_function("gen_kw_data_get", gen_kw_data_get);
    add_function("gen_kw_data_set", gen_kw_data_set);
    add_function("gen_kw_data_size", gen_kw_data_size);
    add_function("gen_kw_data_has_key", gen_kw_data_has_key);
    add_function("gen_kw_ecl_write", gen_kw_ecl_write);
    add_function("gen_kw_get_name", gen_kw_get_name);
    // Summary
    add_function("summary_alloc", summary_alloc, unbound);
    add_function("summary_free", summary_free);
    add_function("summary_get", summary_get);
    add_function("summary_set", summary_set);
    add_function("summary_length", summary_length);
    add_function("summary_undefined_value", summary_undefined_value, unbound);
    // BlockDataConfig
    // BlockObservation
    add_function("block_obs_alloc", block_obs_alloc, unbound);
    add_function("block_obs_free", block_obs_free);
    add_function("block_obs_iget_i", block_obs_iget_i);
    add_function("block_obs_iget_j", block_obs_iget_j);
    add_function("block_obs_iget_k", block_obs_iget_k);
    add_function("block_obs_get_size", block_obs_get_size);
    add_function("block_obs_iget_std", block_obs_iget_std);
    add_function("block_obs_iget_std_scaling", block_obs_iget_std_scaling);
    add_function("block_obs_update_std_scale", block_obs_update_std_scale);
    add_function("block_obs_iget_value", block_obs_iget_value);
    add_function("block_obs_iget_depth", block_obs_iget_depth);
    add_function("block_obs_append_field_obs", block_obs_append_field_obs);
    add_function("block_obs_append_summary_obs", block_obs_append_summary_obs);
    add_function("block_obs_iget_data", block_obs_iget_data);
    // GenObservation
    add_function("gen_obs_alloc__", gen_obs_alloc__, unbound);
    add_function("gen_obs_free", gen_obs_free);
    add_function("gen_obs_load_observation", gen_obs_load_observation);
    add_function("gen_obs_set_scalar", gen_obs_set_scalar);
    add_function("gen_obs_iget_std", gen_obs_iget_std);
    add_function("gen_obs_iget_value", gen_obs_iget_value);
    add_function("gen_obs_iget_std_scaling", gen_obs_iget_std_scaling);
    add_function("gen_obs_get_size", gen_obs_get_size);
    add_function("gen_obs_get_obs_index", gen_obs_get_obs_index);
    add_function("gen_obs_load_data_index", gen_obs_load_data_index);
    add_function("gen_obs_attach_data_index", gen_obs_attach_data_index);
    add_function("gen_obs_update_std_scale", gen_obs_update_std_scale);
    // add_function("gen_obs_load_values", gen_obs_load_values);
    // add_function("gen_obs_load_std", gen_obs_load_std);
    // ObsVector
    add_function("obs_vector_alloc", obs_vector_alloc, unbound);
    add_function("obs_vector_free", obs_vector_free);
    add_function("obs_vector_get_state_kw", obs_vector_get_state_kw);
    add_function("obs_vector_get_key", obs_vector_get_key);
    add_function("obs_vector_iget_node", obs_vector_iget_node);
    add_function("obs_vector_get_num_active", obs_vector_get_num_active);
    add_function("obs_vector_iget_active", obs_vector_iget_active);
    add_function("obs_vector_get_impl_type", obs_vector_get_impl_type);
    add_function("obs_vector_install_node", obs_vector_install_node);
    add_function("obs_vector_get_next_active_step",
                 obs_vector_get_next_active_step);
    add_function("obs_vector_has_data", obs_vector_has_data);
    add_function("obs_vector_get_config_node", obs_vector_get_config_node,
                 return_ref);
    add_function("obs_vector_total_chi2", obs_vector_total_chi2);
    add_function("obs_vector_get_obs_key", obs_vector_get_obs_key);
    add_function("obs_vector_alloc_local_node", obs_vector_alloc_local_node);
    // SummaryObservation
    add_function("summary_obs_alloc", summary_obs_alloc, unbound);
    add_function("summary_obs_free", summary_obs_free);
    add_function("summary_obs_get_value", summary_obs_get_value);
    add_function("summary_obs_get_std", summary_obs_get_std);
    add_function("summary_obs_get_std_scaling", summary_obs_get_std_scaling);
    add_function("summary_obs_get_summary_key", summary_obs_get_summary_key);
    add_function("summary_obs_update_std_scale", summary_obs_update_std_scale);
    add_function("summary_obs_set_std_scale", summary_obs_set_std_scale);
    // EnsemblePlotGenDataVector
    add_function("enkf_plot_genvector_get_size", enkf_plot_genvector_get_size);
    add_function("enkf_plot_genvector_iget", enkf_plot_genvector_iget);
    // EnsemblePlotGenKWVector
    add_function("enkf_plot_gen_kw_vector_get_size",
                 enkf_plot_gen_kw_vector_get_size);
    add_function("enkf_plot_gen_kw_vector_iget", enkf_plot_gen_kw_vector_iget);
    // EnsemblePlotData
    add_function("enkf_plot_data_alloc", enkf_plot_data_alloc, unbound);
    add_function("enkf_plot_data_load", enkf_plot_data_load);
    add_function("enkf_plot_data_get_size", enkf_plot_data_get_size);
    add_function("enkf_plot_data_iget", enkf_plot_data_iget, return_ref);
    add_function("enkf_plot_data_free", enkf_plot_data_free);
    // EnsemblePlotDataVector
    add_function("enkf_plot_tvector_size", enkf_plot_tvector_size);
    add_function("enkf_plot_tvector_iget_value", enkf_plot_tvector_iget_value);
    add_function("enkf_plot_tvector_iget_time", enkf_plot_tvector_iget_time);
    add_function("enkf_plot_tvector_iget_active",
                 enkf_plot_tvector_iget_active);
    // EnsemblePlotGenData
    add_function("enkf_plot_gendata_alloc", enkf_plot_gendata_alloc, unbound);
    add_function("enkf_plot_gendata_get_size", enkf_plot_gendata_get_size);
    add_function("enkf_plot_gendata_load", enkf_plot_gendata_load);
    add_function("enkf_plot_gendata_iget", enkf_plot_gendata_iget, return_ref);
    add_function("enkf_plot_gendata_get_min_values",
                 enkf_plot_gendata_get_min_values, return_ref);
    add_function("enkf_plot_gendata_get_max_values",
                 enkf_plot_gendata_get_max_values, return_ref);
    add_function("enkf_plot_gendata_free", enkf_plot_gendata_free);
    // EnsemblePlotGenKW
    add_function("enkf_plot_gen_kw_alloc", enkf_plot_gen_kw_alloc, unbound);
    add_function("enkf_plot_gen_kw_get_size", enkf_plot_gen_kw_get_size);
    add_function("enkf_plot_gen_kw_load", enkf_plot_gen_kw_load);
    add_function("enkf_plot_gen_kw_iget", enkf_plot_gen_kw_iget, return_ref);
    add_function("enkf_plot_gen_kw_iget_key", enkf_plot_gen_kw_iget_key);
    add_function("enkf_plot_gen_kw_get_keyword_count",
                 enkf_plot_gen_kw_get_keyword_count);
    add_function("enkf_plot_gen_kw_should_use_log_scale",
                 enkf_plot_gen_kw_should_use_log_scale);
    add_function("enkf_plot_gen_kw_free", enkf_plot_gen_kw_free);
    // TimeMap
    add_function("time_map_alloc", time_map_alloc, unbound);
    add_function("time_map_fread", time_map_fread);
    add_function("time_map_fwrite", time_map_fwrite);
    add_function("time_map_fscanf", time_map_fscanf);
    add_function("time_map_iget_sim_days", time_map_iget_sim_days);
    add_function("time_map_iget", time_map_iget);
    add_function("time_map_get_size", time_map_get_size);
    add_function("time_map_try_update", time_map_try_update);
    add_function("time_map_is_strict", time_map_is_strict);
    add_function("time_map_set_strict", time_map_set_strict);
    add_function("time_map_lookup_time", time_map_lookup_time);
    add_function("time_map_lookup_time_with_tolerance",
                 time_map_lookup_time_with_tolerance);
    add_function("time_map_lookup_days", time_map_lookup_days);
    add_function("time_map_get_last_step", time_map_get_last_step);
    add_function("time_map_summary_upgrade107", time_map_summary_upgrade107);
    add_function("time_map_free", time_map_free);
    // Driver
    add_function("queue_driver_alloc", queue_driver_alloc, unbound);
    add_function("queue_driver_free", queue_driver_free);
    add_function("queue_driver_set_option", queue_driver_set_option);
    add_function("queue_driver_get_option", queue_driver_get_option);
    add_function("queue_driver_free_job", queue_driver_free_job);
    add_function("queue_driver_get_status", queue_driver_get_status);
    add_function("queue_driver_kill_job", queue_driver_kill_job);
    add_function("queue_driver_get_max_running", queue_driver_get_max_running);
    add_function("queue_driver_set_max_running", queue_driver_set_max_running);
    add_function("queue_driver_get_name", queue_driver_get_name);
    // EnvironmentVarlist
    add_function("env_varlist_alloc", env_varlist_alloc, unbound);
    add_function("env_varlist_free", env_varlist_free);
    add_function("env_varlist_setenv", env_varlist_setenv);
    add_function("env_varlist_get_size", env_varlist_get_size);
    // ExtJob
    add_function("ext_job_fscanf_alloc", ext_job_fscanf_alloc, unbound);
    add_function("ext_job_free", ext_job_free);
    add_function("ext_job_get_help_text", ext_job_get_help_text);
    add_function("ext_job_get_name", ext_job_get_name);
    add_function("ext_job_set_private_args_from_string",
                 ext_job_set_private_args_from_string);
    add_function("ext_job_is_private", ext_job_is_private);
    add_function("ext_job_get_config_file", ext_job_get_config_file);
    add_function("ext_job_set_config_file", ext_job_set_config_file);
    add_function("ext_job_get_stdin_file", ext_job_get_stdin_file);
    add_function("ext_job_set_stdin_file", ext_job_set_stdin_file);
    add_function("ext_job_get_stdout_file", ext_job_get_stdout_file);
    add_function("ext_job_set_stdout_file", ext_job_set_stdout_file);
    add_function("ext_job_get_stderr_file", ext_job_get_stderr_file);
    add_function("ext_job_set_stderr_file", ext_job_set_stderr_file);
    add_function("ext_job_get_target_file", ext_job_get_target_file);
    add_function("ext_job_set_target_file", ext_job_set_target_file);
    add_function("ext_job_get_executable", ext_job_get_executable);
    add_function("ext_job_set_executable", ext_job_set_executable);
    add_function("ext_job_get_error_file", ext_job_get_error_file);
    add_function("ext_job_get_start_file", ext_job_get_start_file);
    add_function("ext_job_get_max_running", ext_job_get_max_running);
    add_function("ext_job_set_max_running", ext_job_set_max_running);
    add_function("ext_job_get_max_running_minutes",
                 ext_job_get_max_running_minutes);
    add_function("ext_job_set_max_running_minutes",
                 ext_job_set_max_running_minutes);
    add_function("ext_job_get_min_arg", ext_job_get_min_arg);
    add_function("ext_job_get_max_arg", ext_job_get_max_arg);
    add_function("ext_job_iget_argtype", ext_job_iget_argtype);
    add_function("ext_job_get_environment", ext_job_get_environment,
                 return_ref);
    add_function("ext_job_add_environment", ext_job_add_environment);
    add_function("ext_job_get_license_path", ext_job_get_license_path);
    add_function("ext_job_get_arglist", ext_job_get_arglist, return_ref);
    add_function("ext_job_set_args", ext_job_set_args);
    add_function("ext_job_get_argvalues", ext_job_get_argvalues, return_ref);
    add_function("ext_job_clear_environment", ext_job_clear_environment);
    add_function("ext_job_save", ext_job_save);
    // ExtJoblist
    add_function("ext_joblist_alloc", ext_joblist_alloc, unbound);
    add_function("ext_joblist_free", ext_joblist_free);
    add_function("ext_joblist_alloc_list", ext_joblist_alloc_list, return_ref);
    add_function("ext_joblist_get_job", ext_joblist_get_job, return_ref);
    add_function("ext_joblist_del_job", ext_joblist_del_job);
    add_function("ext_joblist_has_job", ext_joblist_has_job);
    add_function("ext_joblist_add_job", ext_joblist_add_job);
    add_function("ext_joblist_get_jobs", ext_joblist_get_jobs, return_ref);
    add_function("ext_joblist_get_size", ext_joblist_get_size);
    // ForwardModel
    add_function("forward_model_alloc", forward_model_alloc, unbound);
    add_function("forward_model_free", forward_model_free);
    add_function("forward_model_clear", forward_model_clear);
    add_function("forward_model_add_job", forward_model_add_job, return_ref);
    add_function("forward_model_alloc_joblist", forward_model_alloc_joblist);
    add_function("forward_model_iget_job", forward_model_iget_job, return_ref);
    add_function("forward_model_get_length", forward_model_get_length);
    add_function("forward_model_formatted_fprintf",
                 forward_model_formatted_fprintf);
    // Job
    // JobQueueNode
    add_function("job_queue_node_alloc_python", job_queue_node_alloc_python,
                 unbound);
    add_function("job_queue_node_free", job_queue_node_free);
    add_function("job_queue_node_submit_simple", job_queue_node_submit_simple);
    add_function("job_queue_node_kill_simple", job_queue_node_kill_simple);
    add_function("job_queue_node_get_status", job_queue_node_get_status);
    add_function("job_queue_node_update_status_simple",
                 job_queue_node_update_status_simple);
    add_function("job_queue_node_set_status", job_queue_node_set_status);
    add_function("job_queue_node_get_submit_attempt",
                 job_queue_node_get_submit_attempt);
    // JobQueue
    add_function("job_queue_alloc", job_queue_alloc, unbound);
    add_function("job_queue_start_user_exit", job_queue_start_user_exit);
    add_function("job_queue_get_user_exit", job_queue_get_user_exit);
    add_function("job_queue_free", job_queue_free);
    add_function("job_queue_set_max_job_duration",
                 job_queue_set_max_job_duration);
    add_function("job_queue_get_max_job_duration",
                 job_queue_get_max_job_duration);
    add_function("job_queue_set_driver", job_queue_set_driver);
    add_function("job_queue_kill_job", job_queue_kill_job);
    add_function("job_queue_run_jobs_threaded", job_queue_run_jobs_threaded);
    add_function("job_queue_iget_driver_data", job_queue_iget_driver_data);
    add_function("job_queue_get_num_running", job_queue_get_num_running);
    add_function("job_queue_get_num_complete", job_queue_get_num_complete);
    add_function("job_queue_get_num_waiting", job_queue_get_num_waiting);
    add_function("job_queue_get_num_pending", job_queue_get_num_pending);
    add_function("job_queue_is_running", job_queue_is_running);
    add_function("job_queue_submit_complete", job_queue_submit_complete);
    add_function("job_queue_iget_sim_start", job_queue_iget_sim_start);
    add_function("job_queue_get_active_size", job_queue_get_active_size);
    add_function("job_queue_set_pause_on", job_queue_set_pause_on);
    add_function("job_queue_set_pause_off", job_queue_set_pause_off);
    add_function("job_queue_get_max_submit", job_queue_get_max_submit);
    add_function("job_queue_iget_job_status", job_queue_iget_job_status);
    add_function("job_queue_get_ok_file", job_queue_get_ok_file);
    add_function("job_queue_get_exit_file", job_queue_get_exit_file);
    add_function("job_queue_get_status_file", job_queue_get_status_file);
    add_function("job_queue_add_job_node", job_queue_add_job_node);
    // Workflow
    add_function("workflow_alloc", workflow_alloc, unbound);
    add_function("workflow_free", workflow_free);
    add_function("workflow_size", workflow_size);
    add_function("workflow_iget_job", workflow_iget_job, return_ref);
    add_function("workflow_iget_arguments", workflow_iget_arguments,
                 return_ref);
    add_function("workflow_try_compile", workflow_try_compile);
    add_function("workflow_get_last_error", workflow_get_last_error,
                 return_ref);
    add_function("worflow_get_src_file", worflow_get_src_file);
    // WorkflowJob
    add_function("workflow_job_alloc", workflow_job_alloc, unbound);
    add_function("workflow_job_alloc_config", workflow_job_alloc_config,
                 unbound);
    add_function("workflow_job_config_alloc", workflow_job_config_alloc,
                 unbound);
    add_function("workflow_job_free", workflow_job_free);
    add_function("workflow_job_get_name", workflow_job_get_name);
    add_function("workflow_job_internal", workflow_job_internal);
    add_function("workflow_job_is_internal_script",
                 workflow_job_is_internal_script);
    add_function("workflow_job_get_internal_script_path",
                 workflow_job_get_internal_script_path);
    add_function("workflow_job_get_function", workflow_job_get_function);
    add_function("workflow_job_get_module", workflow_job_get_module);
    add_function("workflow_job_get_executable", workflow_job_get_executable);
    add_function("workflow_job_get_min_arg", workflow_job_get_min_arg);
    add_function("workflow_job_get_max_arg", workflow_job_get_max_arg);
    add_function("workflow_job_iget_argtype", workflow_job_iget_argtype);
    // WorkflowJoblist
    add_function("workflow_joblist_alloc", workflow_joblist_alloc, unbound);
    add_function("workflow_joblist_free", workflow_joblist_free);
    add_function("workflow_joblist_add_job", workflow_joblist_add_job);
    add_function("workflow_joblist_add_job_from_file",
                 workflow_joblist_add_job_from_file);
    add_function("workflow_joblist_has_job", workflow_joblist_has_job);
    add_function("workflow_joblist_get_job", workflow_joblist_get_job,
                 return_ref);
    // History
    add_function("history_alloc_from_refcase", history_alloc_from_refcase,
                 unbound);
    add_function("history_get_source_string", history_get_source_string,
                 unbound);
    add_function("history_free", history_free);
    // Matrix
    add_function("matrix_alloc", matrix_alloc, unbound);
    add_function("matrix_alloc_identity", matrix_alloc_identity, unbound);
    add_function("matrix_alloc_transpose", matrix_alloc_transpose);
    add_function("matrix_inplace_transpose", matrix_inplace_transpose);
    add_function("matrix_alloc_copy", matrix_alloc_copy);
    add_function("matrix_alloc_sub_copy", matrix_alloc_sub_copy);
    add_function("matrix_free", matrix_free);
    add_function("matrix_iget", matrix_iget);
    add_function("matrix_iset", matrix_iset);
    add_function("matrix_scalar_set", matrix_scalar_set);
    add_function("matrix_scale_column", matrix_scale_column);
    add_function("matrix_scale_row", matrix_scale_row);
    add_function("matrix_copy_column", matrix_copy_column, unbound);
    add_function("matrix_get_rows", matrix_get_rows);
    add_function("matrix_get_columns", matrix_get_columns);
    add_function("matrix_equal", matrix_equal);
    add_function("matrix_pretty_print", matrix_pretty_print);
    // add_function("matrix_fprintf", matrix_fprintf);
    add_function("matrix_random_init", matrix_random_init);
    add_function("matrix_dump_csv", matrix_dump_csv);
    add_function("matrix_alloc_matmul", matrix_alloc_matmul, unbound);
    // PathFormat
    add_function("path_fmt_alloc_directory_fmt", path_fmt_alloc_directory_fmt,
                 unbound);
    add_function("path_fmt_get_fmt", path_fmt_get_fmt);
    add_function("path_fmt_free", path_fmt_free);
    // SubstitutionList
    add_function("subst_list_alloc", subst_list_alloc, unbound);
    add_function("subst_list_free", subst_list_free);
    add_function("subst_list_get_size", subst_list_get_size);
    add_function("subst_list_iget_key", subst_list_iget_key);
    add_function("subst_list_get_value", subst_list_get_value);
    add_function("subst_list_has_key", subst_list_has_key);
    add_function("subst_list_get_doc_string", subst_list_get_doc_string);
    add_function("subst_list_append_copy", subst_list_append_copy);
    // UIReturn
    add_function("ui_return_alloc", ui_return_alloc, unbound);
    add_function("ui_return_free", ui_return_free);
    add_function("ui_return_get_status", ui_return_get_status);
    add_function("ui_return_get_help", ui_return_get_help);
    add_function("ui_return_add_help", ui_return_add_help);
    add_function("ui_return_add_error", ui_return_add_error);
    add_function("ui_return_get_error_count", ui_return_get_error_count);
    add_function("ui_return_get_last_error", ui_return_get_last_error);
    add_function("ui_return_get_first_error", ui_return_get_first_error);
    add_function("ui_return_iget_error", ui_return_iget_error);
}
