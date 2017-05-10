set(TEST_SOURCES
    __init__.py
    test_enkf.py
    test_job_queue.py
    test_sched.py
    test_rms.py
    test_analysis.py
    test_config.py
)

add_python_package("python.tests.legacy.ert"  ${PYTHON_INSTALL_PREFIX}/tests/legacy "${TEST_SOURCES}" False)

addPythonTest(tests.legacy.test_enkf.ErtLegacyEnkfTest)
addPythonTest(tests.legacy.test_job_queue.ErtLegacyJobQueueTest)
addPythonTest(tests.legacy.test_sched.ErtLegacySchedTest)
addPythonTest(tests.legacy.test_rms.ErtLegacyRmsTest)
addPythonTest(tests.legacy.test_analysis.ErtLegacyAnalysisTest)
addPythonTest(tests.legacy.test_config.ErtLegacyConfigTest)
