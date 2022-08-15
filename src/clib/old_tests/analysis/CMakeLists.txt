foreach(name ies_enkf_config ies_enkf_data)
  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} ies/test_${name}.cpp)
  target_link_libraries(${name} res)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

add_executable(ies_module ${TESTS_EXCLUDE_FROM_ALL}
                          ies/test_ies_enkf_module.cpp)
target_link_libraries(ies_module res)
add_test(NAME ies_module COMMAND ies_module
                                 "${TEST_DATA_DIR}/analysis/ies/poly")

add_executable(ies_linalg ${TESTS_EXCLUDE_FROM_ALL} ies/test_ies_linalg.cpp)
target_link_libraries(ies_linalg res)
add_test(NAME ies_linalg COMMAND ies_linalg
                                 "${TEST_DATA_DIR}/analysis/ies/poly")

add_executable(ies_std_compare ${TESTS_EXCLUDE_FROM_ALL}
                               ies/test_ies_std_compare.cpp)
target_link_libraries(ies_std_compare res)
add_test(NAME ies_std_compare COMMAND ies_std_compare
                                      "${TEST_DATA_DIR}/analysis/ies/poly")

add_executable(ies_iteration ${TESTS_EXCLUDE_FROM_ALL}
                             ies/test_ies_iteration.cpp)
target_link_libraries(ies_iteration res)
add_test(NAME ies_iteration COMMAND ies_iteration
                                    "${TEST_DATA_DIR}/analysis/ies/poly_normal")
