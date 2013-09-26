set(PYTHON_SOURCES
    __init__.py
    enkf_assimilation.py
    ensemble_experiment.py
    iterated_smoother.py
    one_more_iteration.py
    simulation_mode.py
    smoother.py
)

add_python_package("Python ert_gui.models.connectors.run"  ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors/run "${PYTHON_SOURCES}" True)

