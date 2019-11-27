set(PYTHON_SOURCES
    __init__.py
    arg_pack.py
    cthread_pool.py
    log.py
    matrix.py
    path_format.py
    res_log.py
    res_version.py
    stat.py
    subprocess.py
    substitution_list.py
    ui_return.py
)

add_python_package("python.res.util"  ${PYTHON_INSTALL_PREFIX}/res/util "${PYTHON_SOURCES}" True)

add_subdirectory(enums)
