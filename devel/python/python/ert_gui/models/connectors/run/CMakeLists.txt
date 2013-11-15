set(PYTHON_SOURCES
    __init__.py
    active_realizations_model.py
    analysis_module_model.py
    ensemble_experiment.py
    ensemble_smoother.py
    simulation_mode.py
    simulations_tracker.py
    target_case_model.py
    workflows_model.py
)

add_python_package("python.ert_gui.models.connectors.run"  ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors/run "${PYTHON_SOURCES}" True)

