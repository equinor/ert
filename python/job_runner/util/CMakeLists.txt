set(PYTHON_SOURCES
    __init__.py
)

add_python_package("python.job_runner.util"  ${PYTHON_INSTALL_PREFIX}/job_runner/util "${PYTHON_SOURCES}" True)
