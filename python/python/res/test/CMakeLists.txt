set(PYTHON_SOURCES
    __init__.py
    ert_test_context.py
)

add_python_package("python.res.test"  ${PYTHON_INSTALL_PREFIX}/res/test "${PYTHON_SOURCES}" True)
