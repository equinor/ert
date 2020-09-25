set(PYTHON_SOURCES
    __init__.py
    history.py
    history_source_enum.py
)

add_python_package("python.res.sched"  ${PYTHON_INSTALL_PREFIX}/res/sched "${PYTHON_SOURCES}" True)

