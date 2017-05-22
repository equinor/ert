project(res-applications C)

if (NOT BUILD_APPLICATIONS)
    return()
endif()

add_executable(rms_extract rms/rms_extract.c)
add_executable(rms_setname rms/rms_setname.c)
add_executable(rms_stat    rms/rms_stat.c)
add_executable(rms_test    rms/rms_test.c)
add_executable(tag_list    rms/tag_list.c)

target_link_libraries(rms_extract rms)
target_link_libraries(rms_setname rms)
target_link_libraries(rms_stat    rms)
target_link_libraries(rms_test    rms)
target_link_libraries(tag_list    rms)

install(TARGETS rms_extract
                rms_setname
                rms_stat
                rms_test
                tag_list
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(ert_module_test analysis/ert_module_test.c)
target_link_libraries(ert_module_test analysis)
install(TARGETS ert_module_test RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (BUILD_TESTS)
    add_test(NAME analysis_module_test_RML
             COMMAND ert_module_test $<TARGET_FILE:rml_enkf>)
endif ()

if (LSF_LIBRARY)
   add_executable(block_node job_queue/block_node.c)
   target_link_libraries(block_node job_queue)
   install(TARGETS block_node RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_executable(sched_summary.x sched/sched_summary.c)
target_link_libraries(sched_summary.x sched ecl)
install(TARGETS sched_summary.x RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
