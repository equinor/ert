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
    test_numpy_and_pandas.py
    test_summary_collector.py
    test_summary_observation_collector.py
)

add_python_package("python.tests.res.enkf.export" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/export "${TEST_SOURCES}" False)

python_config_test(tests.res.enkf.export.test_custom_kw_collector.CustomKwCollectorTest)
python_config_test(tests.res.enkf.export.test_design_matrix.DesignMatrixTest)
python_config_test(tests.res.enkf.export.test_export_join.ExportJoinTest)
python_config_test(tests.res.enkf.export.test_gen_data_collector.GenDataCollectorTest)
python_config_test(tests.res.enkf.export.test_gen_data_observation_collector.GenDataObservationCollectorTest)
python_config_test(tests.res.enkf.export.test_gen_kw_collector.GenKwCollectorTest)
python_config_test(tests.res.enkf.export.test_numpy_and_pandas.NumpyAndPandasTest)
python_config_test(tests.res.enkf.export.test_misfit_collector.MisfitCollectorTest)
python_config_test(tests.res.enkf.export.test_summary_collector.SummaryCollectorTest)
python_config_test(tests.res.enkf.export.test_summary_observation_collector.SummaryObservationCollectorTest)

python_config_test(tests.res.enkf.export.test_arg_loader.ArgLoaderTest)
