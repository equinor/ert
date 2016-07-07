set(PYTHON_SOURCES
    __init__.py
    active_realizations_model.py
    target_case_format_model.py
    target_case_model.py
)

add_python_package("python.ert_gui.models.connectors.run" ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors/run "${PYTHON_SOURCES}" True)

