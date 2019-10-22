set(PYTHON_SOURCES
    __init__.py
)

add_subdirectory(scaling)

add_python_package("python.res.enkf.jobs" ${PYTHON_INSTALL_PREFIX}/res/enkf/jobs "${PYTHON_SOURCES}" True)
