set(TEST_SOURCES
    __init__.py
    test_ert_plugin.py
    test_ert_script.py
    test_ext_job.py
    test_function_ert_script.py
    test_job_queue.py
    test_forward_model_formatted_print.py
    test_workflow.py
    test_workflow_job.py
    test_workflow_joblist.py
    test_workflow_runner.py
    test_jobmanager.py
    test_job_manager_runtime_kw.py
    test_equinor_jobmanager.py
    workflow_common.py
)

add_python_package("python.tests.res.job_queue" ${PYTHON_INSTALL_PREFIX}/tests/res/job_queue "${TEST_SOURCES}" False)
