set(PYTHON_SOURCES
    __init__.py
    arg_loader.py
    design_matrix_reader.py
    gen_data_collector.py
    gen_data_observation_collector.py
    gen_kw_collector.py
    misfit_collector.py
    summary_collector.py
    summary_observation_collector.py
)

add_python_package("python.res.enkf.export"  ${PYTHON_INSTALL_PREFIX}/res/enkf/export "${PYTHON_SOURCES}" True)

