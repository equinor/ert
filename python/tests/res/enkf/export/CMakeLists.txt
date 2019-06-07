set(TEST_SOURCES
    __init__.py
    test_arg_loader.py
    test_custom_kw_collector.py
    test_design_matrix.py
    test_export_join.py
    test_gen_data_collector.py
    test_gen_data_observation_collector.py
    test_gen_kw_collector.py
    test_misfit_collector.py
    test_summary_collector.py
    test_summary_observation_collector.py
)

add_python_package("python.tests.res.enkf.export" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/export "${TEST_SOURCES}" False)
