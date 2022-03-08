find_library(VALGRIND NAMES valgr)

foreach(
  name
  ert_util_arg_pack
  ert_util_matrix
  es_testdata
  ert_util_subst_list
  ert_util_subst_list_add_from_string
  ert_util_block_fs
  thread_pool
  res_util_PATH)

  add_executable(${name} test_${name}.cpp)
  target_link_libraries(${name} res)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

if(VALGRIND)
  set(valgrind_cmd valgrind --error-exitcode=1 --tool=memcheck)
  add_test(NAME test_thread_pool COMMAND ${valgrind_cmd} test_thread_pool)
endif()
