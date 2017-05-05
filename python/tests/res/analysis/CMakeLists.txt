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

addPythonTest(tests.res.analysis.test_analysis_module.AnalysisModuleTest)
addPythonTest(tests.res.analysis.test_options_enum.AnalysisOptionsEnumTest)
addPythonTest(tests.res.analysis.test_linalg.LinalgTest)
addPythonTest(tests.res.analysis.test_rml.RMLTest)
addPythonTest(tests.res.analysis.test_std_enkf.StdEnKFTest)
addPythonTest(tests.res.analysis.test_std_enkf_debug.StdEnKFDebugTest)

