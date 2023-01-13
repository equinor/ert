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
  enkf_fs
  enkf_gen_data_config
  enkf_obs_invalid_path
  enkf_obs
  enkf_obs_vector
  obs_vector
  row_scaling
  trans_func)
  add_executable(${test} ${TESTS_EXCLUDE_FROM_ALL} test_${test}.cpp)
  target_link_libraries(${test} ert)
  fix_install_names(${test})
  add_test(NAME ${test} COMMAND ${test})
endforeach()
