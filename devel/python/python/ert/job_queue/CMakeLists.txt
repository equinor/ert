set(PYTHON_INSTALL_PREFIX "/python/python/ert/job_queue/")
add_python_target(pythonjob_queue ${PYTHON_INSTALL_PREFIX} "driver;__init__;job;job_queue;libjob_queue;queue")