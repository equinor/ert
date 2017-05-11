set(TEST_SOURCES
    __init__.py
    ctest_import.py
    ctest_run.py
    import_tester.py
)

set(TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INSTALL_DIRECTORY ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX})

add_python_package("python.tests" "${PYTHON_INSTALL_PREFIX}/tests" "${TEST_SOURCES}" False)

add_subdirectory(global)
add_subdirectory(res)


if (INSTALL_ERT_LEGACY)
   add_subdirectory(legacy)
endif()

configure_file( test_install.in ${EXECUTABLE_OUTPUT_PATH}/test_install @ONLY )
