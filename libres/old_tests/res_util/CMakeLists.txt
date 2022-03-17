find_library(VALGRIND NAMES valgr)

foreach(name
        ert_util_matrix es_testdata ert_util_subst_list
        ert_util_subst_list_add_from_string ert_util_block_fs res_util_PATH)

  add_executable(${name} test_${name}.cpp)
  target_link_libraries(${name} res)
  add_test(NAME ${name} COMMAND ${name})
endforeach()
