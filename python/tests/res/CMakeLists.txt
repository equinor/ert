function(python_config_test test_class)
  addPythonTest(${test_class})
  set_property(TEST ${test_class}
    APPEND
    PROPERTY
       ENVIRONMENT "ERT_SITE_CONFIG=${PROJECT_SOURCE_DIR}/${SHARE_DIR}/site-config")
endfunction()

set(TEST_SOURCES
    __init__.py
)


add_python_package("python.tests.res"  ${PYTHON_INSTALL_PREFIX}/tests/res "${TEST_SOURCES}" False)
addPythonTest(tests.res.util.test_ui_return)

add_subdirectory(analysis)
add_subdirectory(config)
add_subdirectory(enkf)
add_subdirectory(util)
add_subdirectory(job_queue)
add_subdirectory(run)
add_subdirectory(sched)
add_subdirectory(testcase)
add_subdirectory(simulator)
add_subdirectory(fm)
