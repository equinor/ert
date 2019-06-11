set(TEST_SOURCES
    ctest_import.py
    ctest_run.py
    import_tester.py
    test_files_present.py
)

set(TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INSTALL_DIRECTORY ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX})

add_python_package("python.tests" "${PYTHON_INSTALL_PREFIX}/tests" "${TEST_SOURCES}" False)
configure_file(
   __init__.py
   "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/tests/__init__.py"
   @ONLY
)

add_subdirectory(global)
add_subdirectory(res)
add_subdirectory(share)


configure_file( test_install.in ${EXECUTABLE_OUTPUT_PATH}/test_install @ONLY )
