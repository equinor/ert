add_executable(row_scaling ${TESTS_EXCLUDE_FROM_ALL} test_row_scaling.cpp)
target_link_libraries(row_scaling ert)
fix_install_names(row_scaling)
add_test(NAME row_scaling COMMAND row_scaling)
