set(PYTHON_SOURCES
    __init__.py
    enkf/__init__.py
    job_queue/__init__.py
    sched/__init__.py
    analysis/__init__.py
    config/__init__.py
)

add_python_package("python.ert"  ${PYTHON_INSTALL_PREFIX}/ert "${PYTHON_SOURCES}" True)
