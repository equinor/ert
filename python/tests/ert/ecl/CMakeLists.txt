set(TEST_SOURCES
    __init__.py
    test_ecl_type.py    
)

add_python_package("python.tests.ert.ecl" ${PYTHON_INSTALL_PREFIX}/tests/ert/ecl "${TEST_SOURCES}" False)
addPythonTest(ert.ecl.ecl_type tests.ert.ecl.test_ecl_type.EclDataTypeTest)
