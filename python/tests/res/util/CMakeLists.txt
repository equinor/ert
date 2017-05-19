set(TEST_SOURCES
    __init__.py
    test_res_log.py
)

add_python_package("python.tests.res.util" ${PYTHON_INSTALL_PREFIX}/tests/res/util "${TEST_SOURCES}" False)

addPythonTest(tests.res.util.test_res_log.ResLogTest)