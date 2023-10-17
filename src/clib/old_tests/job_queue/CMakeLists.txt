# -----------------------------------------------------------------
# Additional dependencies
# -----------------------------------------------------------------

find_program(SBATCH "sbatch")

file(COPY "${TEST_DATA_DIR}/job_queue/qsub_emulators/"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# -----------------------------------------------------------------
# Public tests
# -----------------------------------------------------------------

foreach(
  name
  job_status
  job_node
  job_lsf_parse_bsub_stdout
  job_lsf
  job_queue_driver
  job_slurm_driver
  job_mock_slurm
  job_torque)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} ert)
  fix_install_names(${name})
  add_test(NAME ${name} COMMAND ${name})
endforeach()

foreach(name job_slurm_submit job_slurm_runtest)
  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} ert)
  fix_install_names(${name})
  if(SBATCH)
    add_test(NAME ${name} COMMAND ${name})
  endif()
endforeach()

foreach(name job_torque_submit job_queue_stress_task job_program_output
             create_file)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} ert)
  fix_install_names(${name})
endforeach()

if(NOT APPLE)
  add_executable(job_list ${TESTS_EXCLUDE_FROM_ALL} test_job_list.cpp)
  target_link_libraries(job_list ert)
  fix_install_names(job_list)

  add_test(NAME job_list COMMAND $<TARGET_FILE:job_list>)
endif()

add_test(
  NAME job_torque_submit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND job_torque_submit dummyparam)
set_property(TEST job_torque_submit PROPERTY ENVIRONMENT “setenv PATH
                                             ${CMAKE_CURRENT_BINARY_DIR}:$PATH”)
