add_executable(sched_history_summary ${TESTS_EXCLUDE_FROM_ALL}
                                     test_sched_history_summary.cpp)
target_link_libraries(sched_history_summary res)

if(NOT EXISTS "${EQUINOR_TEST_DATA_DIR}")
  return()
endif()

add_test(NAME sched_history_summary1
         COMMAND sched_history_summary
                 "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE")

add_test(NAME sched_history_summary2
         COMMAND sched_history_summary
                 "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Snorre/SNORRE")
