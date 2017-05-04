set(TEST_SOURCES
    __init__.py
    test_testcase.py
    test_testarea.py
    test_mini_config.py
)

add_python_package("python.tests.res.testcase" ${PYTHON_INSTALL_PREFIX}/tests/res/testcase "${TEST_SOURCES}" False)

addPythonTest(res.testcase.testcase tests.res.testcase.test_testcase.TestTestCase)
addPythonTest(res.testcase.testarea tests.res.testcase.test_testarea.TestTestArea)
addPythonTest(res.testcase.mini_config tests.res.testcase.test_mini_config.MiniConfigTest)
