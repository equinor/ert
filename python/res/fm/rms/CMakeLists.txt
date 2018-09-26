set(PYTHON_SOURCES
    __init__.py
    rms_config.py
    rms_run.py
)

add_python_package("python.res.fm.rms" ${PYTHON_INSTALL_PREFIX}/res/fm/rms "${PYTHON_SOURCES}" True)

install_example("rms_config.yml" "${PYTHON_INSTALL_PREFIX}/res/fms/rms")
