set(PYTHON_SOURCES
    __init__.py
    driver.py
    ert_plugin.py
    ert_script.py
    ext_job.py
    ext_joblist.py
    external_ert_script.py
    forward_model.py
    forward_model_status.py
    function_ert_script.py
    job.py
    job_status_type_enum.py
    run_status_type_enum.py
    queue.py
    job_queue_manager.py
    workflow.py
    workflow_job.py
    workflow_joblist.py
    workflow_runner.py
    job_manager.py
    environment_varlist.py
    job_queue_node.py
)

add_python_package("python.res.job_queue"  ${PYTHON_INSTALL_PREFIX}/res/job_queue "${PYTHON_SOURCES}" True)
