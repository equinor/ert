set(PYTHON_SOURCES
    __init__.py
    ecl_config.py
    ecl_run.py
    script.py
)

set(_build_path "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
set(_ecl100_config_file "${_build_path}/ecl100_config.yml")
set(_ecl300_config_file "${_build_path}/ecl300_config.yml")

configure_file( ecl100_config.yml "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl/ecl100_config.yml" COPYONLY)
configure_file( ecl300_config.yml "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/fm/ecl/ecl300_config.yml" COPYONLY)

add_python_package("python.res.fm.ecl" ${PYTHON_INSTALL_PREFIX}/res/fm/ecl "${PYTHON_SOURCES}" True)

install_example(${_ecl100_config_file} "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
install_example(${_ecl300_config_file} "${PYTHON_INSTALL_PREFIX}/res/fm/ecl")
