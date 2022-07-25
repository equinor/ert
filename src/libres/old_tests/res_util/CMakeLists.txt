find_library(VALGRIND NAMES valgr)

foreach(name es_testdata ert_util_subst_list
             ert_util_subst_list_add_from_string res_util_PATH)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} res)
  add_test(NAME ${name} COMMAND ${name})
endforeach()
