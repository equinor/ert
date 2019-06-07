set(TEST_SOURCES
    __init__.py
    test_run.py
)

add_python_package("python.tests.res.run" ${PYTHON_INSTALL_PREFIX}/tests/res/run "${TEST_SOURCES}" False)

