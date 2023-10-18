# -----------------------------------------------------------------
# Additional dependencies
# -----------------------------------------------------------------

find_program(SBATCH "sbatch")

file(COPY "${TEST_DATA_DIR}/job_queue/qsub_emulators/"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# -----------------------------------------------------------------
# Public tests
# -----------------------------------------------------------------

foreach(name job_slurm_driver job_mock_slurm)

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

foreach(name job_torque_submit)

  add_executable(${name} ${TESTS_EXCLUDE_FROM_ALL} test_${name}.cpp)
  target_link_libraries(${name} ert)
  fix_install_names(${name})
endforeach()

add_test(
  NAME job_torque_submit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND job_torque_submit dummyparam)
set_property(TEST job_torque_submit PROPERTY ENVIRONMENT “setenv PATH
                                             ${CMAKE_CURRENT_BINARY_DIR}:$PATH”)
