find_library(VALGRIND NAMES valgr)

foreach(name ert_util_subst_list_add_from_string)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} ert)
  fix_install_names(${name})
  add_test(NAME ${name} COMMAND ${name})
endforeach()
