#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ert::detail {
template <typename> struct wrap_traits;

py::object generic_enum_to_cwrap(const char *name, int value);
    py::object generic_struct_to_cwrap(const char *name, const void *value);
int generic_enum_from_cwrap(const char *name, py::object object);
void *generic_struct_from_cwrap(const char *name, py::object object);
} // namespace ert::detail

#define TRAITS(_Type, _Name)                                                   \
    namespace ert::detail {                                                    \
    template <> struct wrap_traits<_Type> {                                    \
        static constexpr auto name = #_Name;                                   \
        static constexpr auto obj_name = #_Name "_obj";                        \
        static constexpr auto ref_name = #_Name "_ref";                        \
    };                                                                         \
    }

#define STRUCT(_Name)                                                          \
    typedef struct _Name##_struct _Name##_type;                                \
    TRAITS(_Name##_type, _Name)

#define ENUM(_Name)                                                            \
    enum _Name : int;                                                          \
    TRAITS(_Name, _Name)

STRUCT(arg_pack)
STRUCT(bool_vector)
STRUCT(double_vector)
STRUCT(ecl_data_type)
STRUCT(ecl_file)
STRUCT(ecl_file_view)
STRUCT(ecl_grav)
STRUCT(ecl_grid)
STRUCT(ecl_kw)
STRUCT(ecl_region)
STRUCT(ecl_rft_file)
STRUCT(ecl_rft)
STRUCT(ecl_rsthead)
STRUCT(ecl_subsidence)
STRUCT(ecl_sum)
STRUCT(ecl_sum_tstep)
STRUCT(ecl_sum_vector)
STRUCT(ert_test)
STRUCT(fault_block_collection)
STRUCT(fault_block_layer)
STRUCT(fault_block)
STRUCT(fortio)
STRUCT(geo_pointset)
STRUCT(geo_polygon_collection)
STRUCT(geo_polygon)
STRUCT(geo_region)
STRUCT(hash)
STRUCT(int_vector)
STRUCT(layer)
STRUCT(matrix)
STRUCT(node_id)
STRUCT(permutation_vector)
STRUCT(rft_cell)
STRUCT(rng)
STRUCT(smspec_node)
STRUCT(string_hash)
STRUCT(stringlist)
STRUCT(surface)
STRUCT(thread_pool)
STRUCT(well_connection)
STRUCT(well_info)
STRUCT(well_segment)
STRUCT(well_state)
STRUCT(well_time_line)

STRUCT(config_error)
STRUCT(job)
STRUCT(job_queue)
STRUCT(driver)
STRUCT(env_varlist)
STRUCT(forward_model)
STRUCT(time_map)
STRUCT(ensemble_plot_gen_kw)
STRUCT(ensemble_plot_gen_kw_vector)
STRUCT(ensemble_plot_gen_data)
STRUCT(ensemble_plot_gen_data_vector)
STRUCT(ensemble_plot_data_vector)
STRUCT(ensemble_plot_data)
STRUCT(summary)
STRUCT(gen_kw)
STRUCT(gen_kw_config)
STRUCT(ext_param)
STRUCT(ext_param_config)
STRUCT(enkf_node)
STRUCT(site_config)
STRUCT(runpath_list)
STRUCT(run_arg)
STRUCT(row_scaling)
STRUCT(row_config)
STRUCT(rng_config)
STRUCT(queue_config)
STRUCT(log_config)
STRUCT(ert_templates)
STRUCT(local_config)
STRUCT(model_config)
STRUCT(hook_workflow)
STRUCT(forward_load_context)
STRUCT(es_update)
STRUCT(ert_template)
STRUCT(ert_run_context)
STRUCT(analysis_module)
STRUCT(enkf_state)
STRUCT(enkf_obs)
STRUCT(enkf_main)
STRUCT(enkf_fs_manager)
STRUCT(analysis_iter_config)
STRUCT(schema_item)
STRUCT(config_settings)
STRUCT(config_path_elm)
STRUCT(config_item)
STRUCT(content_item)
STRUCT(content_node)
STRUCT(obs_data)
STRUCT(obs_block)
STRUCT(meas_data)
STRUCT(meas_block)
STRUCT(local_updatestep)
STRUCT(local_ministep)
STRUCT(local_obsdata)
STRUCT(ert_workflow)
STRUCT(hook_manager)
STRUCT(ens_config)
STRUCT(res_config)
STRUCT(ert_workflow_list)
STRUCT(ecl_config)
STRUCT(analysis_config)
STRUCT(field)
STRUCT(field_config)
STRUCT(state_map)
STRUCT(config_content)
STRUCT(subst_config)
STRUCT(summary_key_set)
STRUCT(summary_key_matcher)
STRUCT(summary_config)
STRUCT(summary_obs)
STRUCT(active_list)
STRUCT(obs_vector)
STRUCT(local_obsdata_node)
STRUCT(enkf_config_node)
STRUCT(gen_obs)
STRUCT(gen_data_config)
STRUCT(gen_data)
STRUCT(block_obs)
STRUCT(block_data_config)
STRUCT(enkf_fs)
STRUCT(ext_job)
STRUCT(ext_joblist)
STRUCT(job_queue_node)
STRUCT(subst_list)
STRUCT(ui_return)
STRUCT(path_fmt)
STRUCT(history)
STRUCT(workflow)
STRUCT(workflow_job)
STRUCT(workflow_joblist)
STRUCT(config_parser)

ENUM(active_mode_enum)
ENUM(analysis_module_options_enum)
ENUM(config_content_type_enum)
ENUM(config_unrecognized_enum)
ENUM(enkf_field_file_format_enum)
ENUM(enkf_fs_type_enum)
ENUM(enkf_init_mode_enum)
ENUM(enkf_obs_impl_type)
ENUM(enkf_run_mode_enum)
ENUM(enkf_truncation_type_enum)
ENUM(enkf_var_type_enum)
ENUM(ert_impl_type_enum)
ENUM(field_type_enum)
ENUM(gen_data_file_format_type)
ENUM(history_source_enum)
ENUM(hook_runtime_enum)
ENUM(job_status_type_enum)
ENUM(job_submit_status_type_enum)
ENUM(load_fail_type)
ENUM(message_level_enum)
ENUM(queue_driver_enum)
ENUM(realisation_state_enum)
ENUM(rng_alg_type_enum)
ENUM(ui_return_status)

#undef STRUCT
#undef ENUM
#undef TRAITS

namespace ert {
template <typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
py::object to_cwrap(T obj) {
    return detail::generic_enum_to_cwrap(detail::wrap_traits<T>::obj_name, obj);
}

template <typename T, typename = std::enable_if_t<std::is_class_v<T>>>
py::object to_cwrap(const T *obj) {
    return detail::generic_struct_to_cwrap(detail::wrap_traits<T>::obj_name, obj);
}

template <typename T, typename = std::enable_if_t<std::is_class_v<T>>>
py::object to_cwrap_ref(const T *obj) {
    return detail::generic_struct_to_cwrap(detail::wrap_traits<T>::ref_name, obj);
}

template <typename T, typename = std::enable_if_t<std::is_enum_v<T>>>
T from_cwrap(py::object obj) {
    return static_cast<T>(detail::generic_enum_from_cwrap(detail::wrap_traits<T>::name, obj));
}

template <typename T, typename = std::enable_if_t<std::is_class_v<T>>>
T *from_cwrap(py::object obj) {
    return reinterpret_cast<T*>(detail::generic_struct_from_cwrap(detail::wrap_traits<T>::name, obj));
}
} // namespace ert
