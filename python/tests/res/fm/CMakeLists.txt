set(TEST_SOURCES
    __init__.py
    test_ecl_config.py
    test_ecl_run.py
)

add_python_package("python.tests.res.fm"  ${PYTHON_INSTALL_PREFIX}/tests/res/fm "${TEST_SOURCES}" False)
addPythonTest(tests.res.fm.test_ecl_config.EclConfigTest)
addPythonTest(tests.res.fm.test_ecl_run.EclRunTest)
