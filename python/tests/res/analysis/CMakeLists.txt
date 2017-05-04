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

addPythonTest(res.analysis.analysis_module tests.res.analysis.test_analysis_module.AnalysisModuleTest)
addPythonTest(res.analysis.enums tests.res.analysis.test_options_enum.AnalysisOptionsEnumTest)
addPythonTest(res.analysis.linalg tests.res.analysis.test_linalg.LinalgTest)
addPythonTest(res.analysis.rml      tests.res.analysis.test_rml.RMLTest)
addPythonTest(res.analysis.std_enkf tests.res.analysis.test_std_enkf.StdEnKFTest)
addPythonTest(res.analysis.std_enkf_debug tests.res.analysis.test_std_enkf_debug.StdEnKFDebugTest)

