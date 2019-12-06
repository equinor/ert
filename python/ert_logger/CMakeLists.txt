set(PYTHON_SOURCES
    __init__.py
)

add_python_package("python.ert_logger"  ${PYTHON_INSTALL_PREFIX}/ert_logger "${PYTHON_SOURCES}" True)
