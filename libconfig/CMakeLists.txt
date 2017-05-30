project(res-config C)

add_library(config src/config_parser.c
                   src/conf.c
                   src/conf_data.c
                   src/config_content.c
                   src/config_content_item.c
                   src/config_content_node.c
                   src/config_error.c
                   src/config_path_elm.c
                   src/config_root_path.c
                   src/config_schema_item.c
                   src/config_settings.c
                   src/conf_util.c
)

target_link_libraries(config PUBLIC ecl)
target_include_directories(config
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

set_target_properties(config PROPERTIES
    VERSION ${ERT_VERSION_MAJOR}.${ERT_VERSION_MINOR}
    SOVERSION ${ERT_VERSION_MAJOR}
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS config
        EXPORT  res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS)
    return ()
endif()

add_executable(config_append_test tests/config_append_test.c)
target_link_libraries(config_append_test config)
add_test(NAME config_append_test
    COMMAND config_append_test
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/append_test)

add_executable(config_node_test tests/config_node_test.c)
target_link_libraries(config_node_test config)
add_test(NAME config_node_test
    COMMAND config_node_test
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/append_test)

add_executable(config_typeOK tests/config_typeOK.c)
target_link_libraries(config_typeOK config)
add_test(NAME config_typeOK
    COMMAND config_typeOK ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/type_testOK)

add_executable(config_typeFail tests/config_typeFail.c)
target_link_libraries(config_typeFail config)
add_test(NAME config_typeFail
    COMMAND config_typeFail
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/type_testFail)

add_executable(config_path_elm tests/config_path_elm.c)
target_link_libraries(config_path_elm config)
add_test(NAME config_path_elm COMMAND config_path_elm)

add_executable(config_root_path tests/config_root_path.c)
target_link_libraries(config_root_path config)
add_test(NAME config_root_path
    COMMAND config_root_path ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)

add_executable(config_include_test tests/config_include_test.c)
target_link_libraries(config_include_test config)
add_test(NAME config_include_test
         COMMAND config_include_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data include_test)

add_executable(config_content_node tests/config_content_node.c)
target_link_libraries(config_content_node config)
add_test(NAME config_content_node COMMAND config_content_node)

add_executable(config_content_item tests/config_content_item.c)
target_link_libraries(config_content_item config)
add_test(NAME config_content_item
    COMMAND config_content_item
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/content_item_test)

add_executable(config_argc tests/config_argc.c)
target_link_libraries(config_argc config)
add_test(NAME config_argc
    COMMAND config_argc
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/argc_OK
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/argc_less
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/argc_more)

add_executable(config_define tests/config_define.c)
target_link_libraries(config_define config)
add_test(NAME config_define
    COMMAND config_define
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/define_test)

add_executable(config_error tests/config_error.c)
target_link_libraries(config_error config)
add_test(NAME config_error COMMAND config_error)

add_executable(config_content tests/config_content.c)
target_link_libraries(config_content config)
add_test(NAME config_content COMMAND config_content)

add_executable(config_config tests/config_config.c)
target_link_libraries(config_config config)
add_test(NAME config_config COMMAND config_config)

add_executable(config_schema_item tests/config_schema_item.c)
target_link_libraries(config_schema_item config)
add_test(NAME config_schema_item COMMAND config_schema_item)
