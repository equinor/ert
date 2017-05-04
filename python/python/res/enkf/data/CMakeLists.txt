set(PYTHON_SOURCES
    __init__.py
    custom_kw.py
    enkf_node.py
    field.py
    gen_data.py
    gen_kw.py
)

add_python_package("python.res.enkf.data" ${PYTHON_INSTALL_PREFIX}/res/enkf/data "${PYTHON_SOURCES}" True)

