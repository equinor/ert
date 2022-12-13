pybind11_add_module(
  _clib
  SHARED
  res_util/memory.cpp
  res_util/file_utils.cpp
  res_util/subst_list.cpp
  res_util/path_fmt.cpp
  res_util/res_env.cpp
  res_util/block_fs.cpp
  python/init.cpp
  python/logging.cpp
  python/ensemble_config.cpp
  python/model_callbacks.cpp
  config/conf_util.cpp
  config/conf.cpp
  config/conf_data.cpp
  config/config_parser.cpp
  config/config_content.cpp
  config/config_path_stack.cpp
  config/config_content_item.cpp
  config/config_content_node.cpp
  config/config_path_elm.cpp
  config/config_schema_item.cpp
  config/config_keywords.cpp
  rms/rms_file.cpp
  rms/rms_tag.cpp
  rms/rms_tagkey.cpp
  rms/rms_type.cpp
  rms/rms_util.cpp
  analysis/update.cpp
  job_queue/job_status.cpp
  job_queue/job_list.cpp
  job_queue/job_node.cpp
  job_queue/job_queue.cpp
  job_queue/job_queue_status.cpp
  job_queue/local_driver.cpp
  job_queue/lsf_driver.cpp
  job_queue/queue_driver.cpp
  job_queue/slurm_driver.cpp
  job_queue/torque_driver.cpp
  job_queue/job_kw_definitions.cpp
  ${lsb}
  enkf/active_list.cpp
  enkf/time_map.cpp
  enkf/block_fs_driver.cpp
  enkf/config_keys.cpp
  enkf/enkf_config_node.cpp
  enkf/enkf_fs.cpp
  enkf/enkf_node.cpp
  enkf/enkf_obs.cpp
  enkf/enkf_serialize.cpp
  enkf/enkf_state.cpp
  enkf/enkf_types.cpp
  enkf/ensemble_config.cpp
  enkf/ext_param_config.cpp
  enkf/field.cpp
  enkf/field_config.cpp
  enkf/field_trans.cpp
  enkf/fs_driver.cpp
  enkf/fs_types.cpp
  enkf/gen_common.cpp
  enkf/gen_data_config.cpp
  enkf/gen_kw_config.cpp
  enkf/gen_obs.cpp
  enkf/obs_vector.cpp
  enkf/row_scaling.cpp
  enkf/state_map.cpp
  enkf/state_map.cpp
  enkf/summary_config.cpp
  enkf/summary_obs.cpp
  enkf/surface_config.cpp
  enkf/trans_func.cpp)

# -----------------------------------------------------------------
# Target: Python C Extension 'ert._clib'
# -----------------------------------------------------------------

target_link_libraries(
  _clib PUBLIC std::filesystem cJSON::cJSON fmt::fmt Eigen3::Eigen
               cppitertools::cppitertools Threads::Threads)
target_include_directories(_clib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                        "${ECL_INCLUDE_DIRS}")

set_target_properties(_clib PROPERTIES CXX_VISIBILITY_PRESET "default")
install(TARGETS _clib LIBRARY DESTINATION src/ert)

# This extra deploy is only for ctest to be able to locate the script, this file
# is otherwise deployed through setup.py.
file(
  COPY "${CMAKE_CURRENT_SOURCE_DIR}/../../ert/_c_wrappers/job_queue/qstat_proxy.sh"
  DESTINATION "${CMAKE_BINARY_DIR}/old_tests/job_queue"
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

# -----------------------------------------------------------------
# Target: 'libert.so' for use in tests
# -----------------------------------------------------------------

if(BUILD_TESTS)
  add_library(ert SHARED $<TARGET_OBJECTS:_clib>)
  target_link_libraries(ert _clib pybind11::embed ecl)
  fix_install_names(ert)
endif()
