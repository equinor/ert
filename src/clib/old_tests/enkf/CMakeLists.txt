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
  rng_manager
  trans_func
  value_export)
  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

# rng_config test is unique in that it requires mocking of logs
add_executable(rng_config ${TESTS_EXCLUDE_FROM_ALL} test_rng_config.cpp
                          ../../tests/logging.cpp)
target_link_libraries(rng_config ert)
add_test(NAME rng_config COMMAND rng_config)

foreach(
  test
  enkf_queue_config
  enkf_gen_obs_load
  enkf_ert_workflow_list
  enkf_main
  enkf_ecl_config
  enkf_ecl_config_config
  gen_kw_logarithmic
  enkf_executable_path
  enkf_ensemble_GEN_PARAM
  gen_kw)

  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
endforeach()

add_config_test(enkf_executable_path enkf_executable_path)

add_config_test(enkf_ensemble_GEN_PARAM enkf_ensemble_GEN_PARAM
                "${TEST_DATA_DIR}/enkf/ensemble/GEN_PARAM")

add_config_test(enkf_queue_config enkf_queue_config)
add_config_test(enkf_main enkf_main "${TEST_DATA_DIR}/enkf/config" rng)
add_config_test(enkf_gen_obs_load enkf_gen_obs_load
                "${LOCAL_TEST_DATA_DIR}/poly_example/poly.ert")
add_config_test(gen_kw gen_kw
                "${TEST_DATA_DIR}/enkf/config/forward/ert/config_GEN_KW_true")
add_config_test(
  gen_kw_logarithmic gen_kw_logarithmic
  "${TEST_DATA_DIR}/enkf/config/gen_kw_logarithmic/config_GEN_KW_logarithmic")

add_config_test(enkf_ecl_config1 enkf_ecl_config)
add_config_test(enkf_ecl_config2 enkf_ecl_config
                "${LOCAL_TEST_DATA_DIR}/snake_oil/refcase/SNAKE_OIL_FIELD")

add_config_test(enkf_ecl_config_config enkf_ecl_config_config
                "${LOCAL_TEST_DATA_DIR}/snake_oil/snake_oil.ert")

foreach(
  test
  enkf_plot_gendata_fs
  enkf_state_manual_load
  enkf_gen_data_config
  enkf_block_obs
  enkf_obs_fs
  enkf_obs_vector_fs
  enkf_plot_data_fs
  enkf_time_map
  enkf_main_fs)

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

add_equinor_test(enkf_block_obs enkf_block_obs
                 "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE.EGRID")

add_equinor_test(enkf_obs_fs enkf_obs_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/obs_testing/config")

add_equinor_test(enkf_obs_vector_fs enkf_obs_vector_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/obs_testing/config")

add_equinor_test(enkf_plot_data_fs enkf_plot_data_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/plotData/config")

add_equinor_test(enkf_time_map1 enkf_time_map)
add_equinor_test(
  enkf_time_map2
  enkf_time_map
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/EXTRA_TSTEP"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/SHORT"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/MISSING_TSTEP")

add_equinor_test(enkf_main_fs enkf_main_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/plotData/config")

add_equinor_test(enkf_plot_gendata_fs enkf_plot_gendata_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/with_GEN_DATA/config")

add_equinor_test(
  enkf_state_manual_load enkf_state_manual_load
  "${EQUINOR_TEST_DATA_DIR}/config/ecl_sum_compatible_true"
  config_ecl_sum_compatible_true)
