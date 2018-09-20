set(PYTHON_SOURCES
    __init__.py
    shell.py
)

add_python_package("python.res.fm.shell" ${PYTHON_INSTALL_PREFIX}/res/fm/shell "${PYTHON_SOURCES}" True)

