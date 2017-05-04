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
    test_statoil_jobmanager.py
    workflow_common.py
)

add_python_package("python.tests.res.job_queue" ${PYTHON_INSTALL_PREFIX}/tests/res/job_queue "${TEST_SOURCES}" False)

addPythonTest(res.job_queue.forward_model_formatted_print
    tests.res.job_queue.test_forward_model_formatted_print.ForwardModelFormattedPrintTest)

addPythonTest(res.job_queue.job_status_type_enum tests.res.job_queue.test_job_queue.JobQueueTest)
addPythonTest(res.job_queue.workflow_job tests.res.job_queue.test_workflow_job.WorkflowJobTest)
addPythonTest(res.job_queue.workflow tests.res.job_queue.test_workflow.WorkflowTest)
addPythonTest(res.job_queue.workflow_joblist tests.res.job_queue.test_workflow_joblist.WorkflowJoblistTest)
addPythonTest(res.job_queue.ert_plugin tests.res.job_queue.test_ert_plugin.ErtPluginTest)
addPythonTest(res.job_queue.ert_script tests.res.job_queue.test_ert_script.ErtScriptTest)
addPythonTest(res.job_queue.function_ert_script tests.res.job_queue.test_function_ert_script.FunctionErtScriptTest)
addPythonTest(res.job_queue.workflow_runner tests.res.job_queue.test_workflow_runner.WorkflowRunnerTest)
addPythonTest(res.job_queue.ext_job tests.res.job_queue.test_ext_job.ExtJobTest)
addPythonTest(res.job_queue.jobmanager tests.res.job_queue.test_jobmanager.JobManagerTest)

if (STATOIL_TESTDATA_ROOT)
    addPythonTest(res.job_queue.jobmanager_statoil
                  tests.res.job_queue.test_statoil_jobmanager.JobManagerStatoilTest
                  LABELS StatoilData)
endif()
