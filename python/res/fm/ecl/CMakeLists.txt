set(PYTHON_SOURCES
    __init__.py
    ecl_config.py
    ecl_run.py
)

add_python_package("python.res.fm.ecl" ${PYTHON_INSTALL_PREFIX}/res/fm/ecl "${PYTHON_SOURCES}" True)

install_example("ecl_config.yml" "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
