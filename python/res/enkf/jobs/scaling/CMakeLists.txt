set(PYTHON_SOURCES
    scaling_job.py
    job_config.py
    measured_data.py
    scaled_matrix.py
    __init__.py
)

add_python_package("python.res.enkf.jobs.scaling" ${PYTHON_INSTALL_PREFIX}/res/enkf/jobs "${PYTHON_SOURCES}" True)
