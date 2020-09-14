set(PYTHON_SOURCES
    __init__.py
    enkf_config_node.py
    field_type_enum.py
    field_config.py
    gen_data_config.py
    gen_kw_config.py
    summary_config.py
    ext_param_config.py
)

add_python_package("python.res.enkf.config" ${PYTHON_INSTALL_PREFIX}/res/enkf/config "${PYTHON_SOURCES}" True)

