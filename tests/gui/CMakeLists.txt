set(TEST_SOURCES
    __init__.py
    test_multiple_data_assimilation.py
)

add_python_package("python.tests.gui"  ${PYTHON_INSTALL_PREFIX}/tests/gui "${TEST_SOURCES}" False)

addPythonTest(tests.gui.test_multiple_data_assimilation.MDAWeightsTest)

add_subdirectory(ertshell)
add_subdirectory(ide)
add_subdirectory(models)
add_subdirectory(plottery)
