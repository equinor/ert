project(res_util C)

if (ERT_HAVE_REGEXP)
 list(APPEND opt_srcs src/template_loop.c)
endif ()

if (HAVE_PTHREAD)
    list(APPEND opt_srcs src/block_fs.c)
endif ()

add_library( res_util src/res_log.c
                      src/subst_list.c
                      src/template.c
                      src/template_loop.c
                      src/block_fs.c
                      src/res_version.c
                      ${opt_srcs})


target_link_libraries( res_util PUBLIC ecl )

target_include_directories(res_util
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

set_target_properties( res_util PROPERTIES VERSION ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR} SOVERSION ${RES_VERSION_MAJOR} )

target_compile_definitions(res_util PRIVATE
            -DGIT_COMMIT=${GIT_COMMIT}
            -DGIT_COMMIT_SHORT=${GIT_COMMIT_SHORT}
            -DRES_VERSION_MAJOR=${RES_VERSION_MAJOR}
            -DRES_VERSION_MINOR=${RES_VERSION_MINOR}
            -DRES_VERSION_MICRO=${RES_VERSION_MICRO}
            -DCOMPILE_TIME_STAMP="${RES_BUILD_TIME}"
)



install(DIRECTORY include/ DESTINATION include)
install(TARGETS res_util
        EXPORT res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS)
    return ()
endif()

#Tests go below
add_executable(ert_util_subst_list tests/ert_util_subst_list.c)
target_link_libraries(ert_util_subst_list res_util)
add_test(NAME ert_util_subst_list COMMAND ert_util_subst_list)

add_executable(ert_util_block_fs tests/ert_util_block_fs.c)
target_link_libraries(ert_util_block_fs res_util)
add_test(NAME ert_util_block_fs COMMAND ert_util_block_fs)

