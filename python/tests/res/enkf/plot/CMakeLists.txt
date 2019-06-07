set(TEST_SOURCES
    __init__.py
    test_plot_data.py
)

add_python_package("python.tests.res.enkf.plot" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/plot "${TEST_SOURCES}" False)
