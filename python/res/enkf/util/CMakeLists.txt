set(PYTHON_SOURCES
    __init__.py
    time_map.py
)

add_python_package("python.res.enkf.util"  ${PYTHON_INSTALL_PREFIX}/res/enkf/util "${PYTHON_SOURCES}" True)

