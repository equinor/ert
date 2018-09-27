set(TEST_SOURCES
    __init__.py
    test_testcase.py
    test_testarea.py
    test_mini_config.py
)

add_python_package("python.tests.res.testcase" ${PYTHON_INSTALL_PREFIX}/tests/res/testcase "${TEST_SOURCES}" False)

python_config_test(tests.res.testcase.test_testcase.TestTestCase)
python_config_test(tests.res.testcase.test_testarea.TestTestArea)
python_config_test(tests.res.testcase.test_mini_config.MiniConfigTest)
