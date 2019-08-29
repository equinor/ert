set(PYTHON_SOURCES
    __init__.py
    file.py
    interactive.py
    network.py
    message.py
)


add_python_package("python.job_runner.reporting"  ${PYTHON_INSTALL_PREFIX}/job_runner/reporting "${PYTHON_SOURCES}" True)
