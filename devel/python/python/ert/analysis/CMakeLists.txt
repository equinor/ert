set(PYTHON_SOURCES
    __init__.py
)

add_python_package("python.ert.analysis"  ${PYTHON_INSTALL_PREFIX}/ert/analysis "${PYTHON_SOURCES}" T)


add_subdirectory(enums)
