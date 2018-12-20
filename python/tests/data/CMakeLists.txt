set(TEST_SOURCES
    __init__.py
    test_data.py
)

add_python_package("python.tests.data" ${PYTHON_INSTALL_PREFIX}/tests/data "${TEST_SOURCES}" False)

addPythonTest(tests.data.test_data.TestData)
