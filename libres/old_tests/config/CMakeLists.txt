macro(config_test name input_file)
  add_test(NAME ${name} COMMAND ${name} "${TEST_DATA_DIR}/config/${input_file}")
endmacro()

foreach(name config_content_node config_path_elm config_error config_content
             config_config config_schema_item)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} res)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

foreach(
  name
  config_append
  config_node
  config_typeOK
  config_typeFail
  config_root_path
  config_include
  config_content_item
  config_define
  config_argc)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} res)
endforeach()

config_test(config_append append_test)
config_test(config_node append_test)
config_test(config_typeOK type_testOK)
config_test(config_typeFail type_testFail)
config_test(config_content_item content_item_test)
config_test(config_define define_test)

add_test(NAME config_root_path COMMAND config_root_path
                                       "${TEST_DATA_DIR}/config")

add_test(NAME config_include COMMAND config_include "${TEST_DATA_DIR}/config"
                                     include_test)

add_test(
  NAME config_argc
  COMMAND
    config_argc "${TEST_DATA_DIR}/config/argc_OK"
    "${TEST_DATA_DIR}/config/argc_less" "${TEST_DATA_DIR}/config/argc_more")
