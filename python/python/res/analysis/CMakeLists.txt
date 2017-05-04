set(PYTHON_SOURCES
    __init__.py
    analysis_module.py
    linalg.py
)

add_python_package("python.res.analysis."  ${PYTHON_INSTALL_PREFIX}/res/analysis/ "${PYTHON_SOURCES}" True)

add_subdirectory(enums)
