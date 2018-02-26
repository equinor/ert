project(res-jobqueue C)

if (LSF_LIBRARY AND LSF_HEADER_PATH)
    set(lsb src/lsb.c)
    install(DIRECTORY include/
            DESTINATION include
            PATTERN lsb.h)
    target_compile_definitions(job_queue PUBLIC -DHAVE_LSF_LIBRARY)
    set(lsf ${LSF_LIBRARY})
else ()
   message(STATUS "LSF not found")
endif ()

add_library(job_queue src/ext_job.c
                      src/ext_joblist.c
                      src/forward_model.c
                      src/job_status.c
                      src/job_list.c
                      src/job_node.c
                      src/job_queue.c
                      src/job_queue_manager.c
                      src/job_queue_status.c
                      src/local_driver.c
                      src/lsf_driver.c
                      src/queue_driver.c
                      src/rsh_driver.c
                      src/torque_driver.c
                      src/workflow.c
                      src/workflow_job.c
                      src/workflow_joblist.c
                      src/environment_varlist.c
                      src/job_kw_definitions.c
                      ${lsb}
)

target_link_libraries(job_queue PUBLIC res_util config ecl ${lsf})
target_include_directories(job_queue
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)
set_target_properties(job_queue PROPERTIES
    VERSION ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR}
    SOVERSION ${RES_VERSION_MAJOR}
)

install(DIRECTORY include/
        DESTINATION include
        PATTERN lsb.h EXCLUDE)
install(TARGETS job_queue
        EXPORT  res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS)
    return ()
endif ()

# job_workflow requires this binary being available
add_executable(create_file tests/create_file.c)
target_link_libraries(create_file job_queue)

add_executable(job_status_test tests/job_status_test.c)
target_link_libraries(job_status_test job_queue pthread)
add_test(NAME job_status_test COMMAND job_status_test)

add_executable(job_loadOK tests/job_loadOK.c)
target_link_libraries(job_loadOK job_queue)
add_test(NAME job_loadOK1
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/internalOK)
add_test(NAME job_loadOK2
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/externalOK)
add_test(NAME job_loadOK3
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/internalOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/externalOK)

add_executable(job_loadFail tests/job_loadFail.c)
target_link_libraries(job_loadFail job_queue)
add_test(NAME job_loadFail1
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/internalFail)
add_test(NAME job_loadFail2
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/externalFail)
add_test(NAME job_loadFail3
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/internalFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/externalFail)

add_executable(job_node_test tests/job_node_test.c)
target_link_libraries(job_node_test job_queue)
add_test(NAME job_node_test COMMAND job_node_test)

add_executable(job_list_test tests/job_list_test.c)
target_link_libraries(job_list_test job_queue)
add_test(NAME job_list_test
         COMMAND valgrind --leak-check=full --error-exitcode=1
                 $<TARGET_FILE:job_list_test>)

add_executable(job_lsf_parse_bsub_stdout tests/job_lsf_parse_bsub_stdout.c)
target_link_libraries(job_lsf_parse_bsub_stdout job_queue)
add_test(NAME job_lsf_parse_bsub_stdout COMMAND job_lsf_parse_bsub_stdout)

add_executable(job_workflow_test tests/job_workflow_test.c)
target_link_libraries(job_workflow_test job_queue)
add_test(NAME job_workflow_test
         COMMAND job_workflow_test
                 ${EXECUTABLE_OUTPUT_PATH}
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/internal_job)

add_executable(job_lsf_exclude_hosts_test tests/job_lsf_exclude_hosts_test.c)
target_link_libraries(job_lsf_exclude_hosts_test job_queue ecl)
add_test(NAME job_lsf_exclude_hosts_test
         COMMAND job_lsf_exclude_hosts_test job_program NULL LOCAL)

add_executable(job_program_output tests/job_program_output.c)
target_link_libraries(job_program_output ecl)

add_executable(job_queue_test tests/job_job_queue_test.c)
target_link_libraries(job_queue_test job_queue)
add_test(NAME job_queue_test
         COMMAND job_queue_test
                 $<TARGET_FILE:job_program_output>)
set_tests_properties(job_queue_test PROPERTIES LABELS "SLOW_1")

add_executable(job_queue_stress_task tests/job_queue_stress_task.c)
target_link_libraries(job_queue_stress_task ecl)

add_executable(job_queue_stress_test tests/job_queue_stress_test.c)
target_link_libraries(job_queue_stress_test job_queue)
add_test(NAME job_queue_stress_test
         COMMAND job_queue_stress_test
                 $<TARGET_FILE:job_queue_stress_task>
                 False)
set_tests_properties(job_queue_stress_test PROPERTIES LABELS "SLOW_1")

add_test(NAME job_queue_user_exit
         COMMAND job_queue_stress_test
                 $<TARGET_FILE:job_queue_stress_task>
                 True)

add_executable(job_queue_timeout_test tests/job_queue_timeout_test.c)
target_link_libraries(job_queue_timeout_test job_queue)
add_test(NAME job_queue_timeout_test
         COMMAND job_queue_timeout_test $<TARGET_FILE:job_queue_stress_task>)

add_executable(job_lsf_test tests/job_lsf_test.c)
target_link_libraries(job_lsf_test job_queue)
add_test(NAME job_lsf_test COMMAND job_lsf_test)

add_executable(job_queue_driver_test tests/job_queue_driver_test.c)
target_link_libraries(job_queue_driver_test job_queue)
add_test(NAME job_queue_driver_test COMMAND job_queue_driver_test)

add_executable(job_torque_test tests/job_torque_test.c)
target_link_libraries(job_torque_test job_queue ecl)
add_test(NAME job_torque_test COMMAND job_torque_test)

add_executable(job_queue_manager tests/job_queue_manager.c)
target_link_libraries(job_queue_manager job_queue ecl)
add_test(NAME job_queue_manager COMMAND job_queue_manager)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qsub_emulators/
     DESTINATION ${EXECUTABLE_OUTPUT_PATH})

add_executable(job_torque_submit_test tests/job_torque_submit_test.c)
target_link_libraries(job_torque_submit_test job_queue)
add_test(NAME job_torque_submit_test
         WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
         COMMAND job_torque_submit_test dummyparam)
set_property(TEST job_torque_submit_test PROPERTY ENVIRONMENT “setenv PATH ${EXECUTABLE_OUTPUT_PATH}:$PATH”)

add_executable(ext_joblist_test tests/ext_joblist_test.c)
target_link_libraries(ext_joblist_test job_queue)
add_test(NAME ext_joblist_test
         COMMAND ext_joblist_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/jobs/util
                 ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT ERT_LSF_SUBMIT_TEST)
    return ()
endif()

# The testing of the lsf submit capabilities is quite troublesome for
# two reasons, and therefor by default disabled:
#
#
#  1. The shell based LSF commands require that user running the
#     bsub/bjobs/bxxx command has passwordless ssh configured to log in
#     to the lsf server. When the build and testing is run as a common
#     'jenkins' user this becomes difficult.
#
#  2. Submitting through the lsf library requires that the build/test
#     server actually is a LIM host; which it typically is not.
#
#-----------------------------------------------------------------
#
# This should be a space separated list of servers which will be
# tried out when testing the LSF submit capability. The test program
# will interpret the special strings 'NULL' and 'LOCAL' as follows:
#
#   NULL:  Submit using the linked in library functions.
#   LOCAL: Submit using shell commands on the current server
#
set(LSF_SERVER "" CACHE STRING  "List of LSF servers for testing")

if (LSF_LIBRARY)
   add_executable(job_lsb_test tests/job_lsb_test.c)
   target_link_libraries(job_lsb_test job_queue util test_util)
   add_test(NAME job_lsb_test COMMAND job_lsb_test)

   add_executable(job_lsb tests/job_lsb.c)
   target_link_libraries(job_lsb job_queue util test_util)
   add_test(NAME job_lsb COMMAND job_lsb)
endif()

add_executable(job_lsf_remote_submit_test tests/job_lsf_remote_submit_test.c)
target_link_libraries(job_lsf_remote_submit_test job_queue util test_util)

add_executable(job_lsf_library_submit_test tests/job_lsf_library_submit_test.c)
target_link_libraries(job_lsf_library_submit_test job_queue util)

add_executable(job_program tests/job_program.c)

if (LSF_SERVER)
    add_test(NAME job_lsf_remote_submit_test
             COMMAND job_lsf_remote_submit_test
                     job_program
                     ${LSF_SERVER}
                     NULL
                     LOCAL)
else()
    add_test(NAME job_lsf_remote_submit_test
             COMMAND job_lsf_remote_submit_test
                     job_program
                     NULL
                     LOCAL)
endif()

install(TARGETS job_program job_lsf_library_submit_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
