set(TEST_SOURCES
    __init__.py
    test_config.py
)

add_python_package("python.tests.res.config" ${PYTHON_INSTALL_PREFIX}/tests/res/config "${TEST_SOURCES}" False)

addPythonTest(tests.res.config.test_config.ConfigTest)
