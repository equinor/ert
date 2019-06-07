set(TEST_SOURCES
    __init__.py
    test_enkf.py
    test_job_queue.py
    test_sched.py
    test_analysis.py
    test_config.py
        )

add_python_package("python.tests.legacy.ert"  ${PYTHON_INSTALL_PREFIX}/tests/legacy "${TEST_SOURCES}" False)
