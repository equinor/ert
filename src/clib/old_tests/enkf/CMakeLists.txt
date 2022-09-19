# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------

function(add_config_test name command)
  add_test(NAME ${name} COMMAND ${command} ${ARGN})

  set_property(TEST ${name} PROPERTY ENVIRONMENT
                                     "ERT_SITE_CONFIG=${SITE_CONFIG_PATH}")
endfunction()

function(add_equinor_test name command)
  add_test(NAME ${name} COMMAND ${command} ${ARGN})

  set_property(TEST ${name} PROPERTY LABELS EquinorData)
endfunction()

# -----------------------------------------------------------------
# Public tests
# -----------------------------------------------------------------

foreach(
  test
  enkf_active_list
  enkf_analysis_config
  enkf_config_node
  enkf_enkf_config_node_gen_data
  enkf_ensemble
  enkf_ensemble_config
  enkf_fs
  enkf_gen_data_config_parse
  enkf_iter_config
  enkf_meas_data
  enkf_model_config
  enkf_obs_invalid_path
  enkf_obs
  enkf_obs_vector
  enkf_plot_data
  enkf_plot_gendata
  enkf_plot_gen_kw
  enkf_plot_gen_kw_vector
  enkf_plot_genvector
  enkf_plot_tvector
  enkf_run_arg
  enkf_state_map
  obs_vector
  measured
  obs_mask_deactivate
  row_scaling
  trans_func
  value_export)
  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

foreach(test
        enkf_queue_config enkf_ert_workflow_list enkf_ecl_config
        enkf_ecl_config_config enkf_executable_path enkf_ensemble_GEN_PARAM)

  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
endforeach()

add_config_test(enkf_executable_path enkf_executable_path)

add_config_test(enkf_ensemble_GEN_PARAM enkf_ensemble_GEN_PARAM
                "${TEST_DATA_DIR}/enkf/ensemble/GEN_PARAM")

add_config_test(enkf_queue_config enkf_queue_config)
add_config_test(enkf_ecl_config1 enkf_ecl_config)
add_config_test(enkf_ecl_config2 enkf_ecl_config
                "${LOCAL_TEST_DATA_DIR}/snake_oil/refcase/SNAKE_OIL_FIELD")

add_config_test(enkf_ecl_config_config enkf_ecl_config_config
                "${LOCAL_TEST_DATA_DIR}/snake_oil/snake_oil.ert")

foreach(test enkf_gen_data_config enkf_time_map)

  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
endforeach()

# -----------------------------------------------------------------
# Equinor tests
# -----------------------------------------------------------------

if(NOT EXISTS "${EQUINOR_TEST_DATA_DIR}")
  return()
endif()

add_equinor_test(
  enkf_gen_data_config enkf_gen_data_config
  "${EQUINOR_TEST_DATA_DIR}/config/gendata_test/RFT_E-3H_21"
  "${EQUINOR_TEST_DATA_DIR}/config/gendata_test/RFT_E-3H_21_empty")

add_equinor_test(enkf_time_map1 enkf_time_map)
add_equinor_test(
  enkf_time_map2
  enkf_time_map
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/EXTRA_TSTEP"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/SHORT"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/MISSING_TSTEP")
