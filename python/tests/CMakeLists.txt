set(TEST_SOURCES
    __init__.py
    ctest_import.py
    ctest_run.py
    import_tester.py
)

set(TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INSTALL_DIRECTORY ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX})

add_python_package("python.tests" "${PYTHON_INSTALL_PREFIX}/tests" "${TEST_SOURCES}" False)

add_subdirectory(res)

add_test( NAME python.tests.res.import
          WORKING_DIRECTORY ${INSTALL_DIRECTORY}
          COMMAND tests/ctest_import.py ${INSTALL_DIRECTORY} "res")

if (ERT_BUILD_GUI)
    add_subdirectory(gui)

    add_test( NAME python.tests.ert_gui.import
              WORKING_DIRECTORY ${INSTALL_DIRECTORY}
              COMMAND tests/ctest_import.py ${INSTALL_DIRECTORY} "ert_gui")
endif()


