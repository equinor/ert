project(res-applications C)

if (NOT BUILD_APPLICATIONS)
    return()
endif()

add_executable(bls_bin block_fs/bls.cpp)
add_executable(bfs_extract_bin block_fs/bfs_extract.cpp)
target_link_libraries(bls_bin res)
target_link_libraries(bfs_extract_bin res)


add_executable(rms_extract_bin rms/rms_extract.c)
add_executable(rms_setname_bin rms/rms_setname.c)
add_executable(rms_stat_bin    rms/rms_stat.c)
add_executable(rms_test_bin    rms/rms_test.c)
add_executable(tag_list_bin    rms/tag_list.c)

target_link_libraries(rms_extract_bin res)
target_link_libraries(rms_setname_bin res)
target_link_libraries(rms_stat_bin    res)
target_link_libraries(rms_test_bin    res)
target_link_libraries(tag_list_bin    res)

install(TARGETS rms_extract_bin
                rms_setname_bin
                rms_stat_bin
                rms_test_bin
                tag_list_bin
                bls_bin
                bfs_extract_bin
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(ert_module_test_bin analysis/ert_module_test.c)
target_link_libraries(ert_module_test_bin res)
install(TARGETS ert_module_test_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (BUILD_TESTS)
    add_test(NAME analysis_module_test_RML
             COMMAND ert_module_test_bin $<TARGET_FILE:rml_enkf>)
endif ()

if (LSF_LIBRARY)
   add_executable(block_node_bin job_queue/block_node.c)
   target_link_libraries(block_node res)
   install(TARGETS block_node_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(sched_summary.x_bin sched/sched_summary.c)
target_link_libraries(sched_summary.x_bin res)
install(TARGETS sched_summary.x_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


foreach(executable 
rms_extract
rms_setname
rms_stat
rms_test
tag_list
bls
bfs_extract
ert_module_test
block_node
sched_summary.x)

set( EXEC ${executable}_bin)
configure_file(wrapper.in  ${EXECUTABLE_OUTPUT_PATH}/${executable} )
install(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${executable} DESTINATION ${CMAKE_INSTALL_BINDIR})

endforeach(executable)
