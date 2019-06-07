set( TEST_SOURCES
     __init__.py
     test_import.py
)

add_python_package("python.tests.global"  "${PYTHON_INSTALL_PREFIX}/tests/global" "${TEST_SOURCES}" False)
