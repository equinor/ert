set(PYTHON_SOURCES
    __init__.py
    block_observation.py
    block_data_config.py
    gen_observation.py
    obs_vector.py
    summary_observation.py
)

add_python_package("python.res.enkf.observations"  ${PYTHON_INSTALL_PREFIX}/res/enkf/observations "${PYTHON_SOURCES}" True)

