set(PYTHON_SOURCES
    __init__.py
    test_run.py
    extended_testcase.py
    source_enumerator.py
    test_area.py
)

add_python_package("python.ert.test"  ${PYTHON_INSTALL_PREFIX}/ert/test "${PYTHON_SOURCES}" True)

