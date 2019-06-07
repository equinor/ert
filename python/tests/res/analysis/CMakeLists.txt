set(TEST_SOURCES
    __init__.py
    test_analysis_module.py
    test_linalg.py
    test_options_enum.py
    test_rml.py
    test_std_enkf.py
    test_std_enkf_debug.py
)

add_python_package("python.tests.res.analysis" ${PYTHON_INSTALL_PREFIX}/tests/res/analysis "${TEST_SOURCES}" False)
