set(PYTHON_SOURCES
    __init__.py
    res_log.py
    substitution_list.py
)

add_python_package("python.res.util"  ${PYTHON_INSTALL_PREFIX}/res/util "${PYTHON_SOURCES}" True)