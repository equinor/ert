set(PYTHON_SOURCES
    __init__.py
    job.py
    runner.py
)

add_python_package("python.job_runner"  ${PYTHON_INSTALL_PREFIX}/job_runner "${PYTHON_SOURCES}" True)

add_subdirectory( io )
add_subdirectory( reporting )
add_subdirectory( util )
