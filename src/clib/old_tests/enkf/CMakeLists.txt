# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------

function(add_config_test name command)
  add_test(NAME ${name} COMMAND ${command} ${ARGN})

  set_property(TEST ${name} PROPERTY ENVIRONMENT
                                     "ERT_SITE_CONFIG=${SITE_CONFIG_PATH}")
endfunction()

# -----------------------------------------------------------------
# Public tests
# -----------------------------------------------------------------

foreach(
  test
  enkf_active_list
  enkf_config_node
  enkf_ensemble
  enkf_ensemble_config
  enkf_fs
  enkf_gen_data_config
  enkf_gen_data_config_parse
  enkf_meas_data
  enkf_model_config
  enkf_obs_invalid_path
  enkf_obs
  enkf_obs_vector
  enkf_plot_gendata
  enkf_plot_gen_kw
  enkf_plot_gen_kw_vector
  enkf_plot_genvector
  enkf_plot_tvector
  enkf_run_arg
  enkf_state_map
  enkf_time_map
  obs_vector
  measured
  obs_mask_deactivate
  row_scaling
  trans_func)
  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

foreach(test enkf_ert_workflow_list)

  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
endforeach()

add_config_test(enkf_time_map2 enkf_time_map
                "${LOCAL_TEST_DATA_DIR}/snake_oil/refcase/SNAKE_OIL_FIELD")
