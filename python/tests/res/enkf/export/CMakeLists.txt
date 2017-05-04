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

addPythonTest(res.enkf.export.custom_kw_collector tests.res.enkf.export.test_custom_kw_collector.CustomKwCollectorTest)
addPythonTest(res.enkf.export.design_matrix tests.res.enkf.export.test_design_matrix.DesignMatrixTest)
addPythonTest(res.enkf.export.export_join tests.res.enkf.export.test_export_join.ExportJoinTest)
addPythonTest(res.enkf.export.gen_data_collector tests.res.enkf.export.test_gen_data_collector.GenDataCollectorTest)
addPythonTest(res.enkf.export.gen_data_observation_collector tests.res.enkf.export.test_gen_data_observation_collector.GenDataObservationCollectorTest)
addPythonTest(res.enkf.export.gen_kw_collector tests.res.enkf.export.test_gen_kw_collector.GenKwCollectorTest)
addPythonTest(res.enkf.export.numpy_and_pandas tests.res.enkf.export.test_numpy_and_pandas.NumpyAndPandasTest)
addPythonTest(res.enkf.export.misfit_collector tests.res.enkf.export.test_misfit_collector.MisfitCollectorTest)
addPythonTest(res.enkf.export.summary_collector tests.res.enkf.export.test_summary_collector.SummaryCollectorTest)
addPythonTest(res.enkf.export.summary_observation_collector tests.res.enkf.export.test_summary_observation_collector.SummaryObservationCollectorTest)

if (STATOIL_TESTDATA_ROOT)
   addPythonTest(res.enkf.export.arg_loader tests.res.enkf.export.test_arg_loader.ArgLoaderTest LABELS StatoilData)
endif()
