set(PYTHON_SOURCES
    __init__.py
)
add_subdirectory(ecl)
add_python_package("python.res.fm" ${PYTHON_INSTALL_PREFIX}/res/fm "${PYTHON_SOURCES}" True)

