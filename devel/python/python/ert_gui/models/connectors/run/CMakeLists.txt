set(PYTHON_SOURCES
    __init__.py
    active_realizations_model.py
    analysis_module_model.py
    analysis_module_variables_model.py
    iterated_analysis_module_model.py
    number_of_iterations.py
    simulations_tracker.py
    target_case_format_model.py
    target_case_model.py
    workflows_model.py
)

add_python_package("python.ert_gui.models.connectors.run" ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors/run "${PYTHON_SOURCES}" True)

