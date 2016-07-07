set(PYTHON_SOURCES
    __init__.py
    basic_model.py
    choice_model.py
    model.py
    path_model.py
)

add_python_package("python.ert_gui.models.mixins"  ${PYTHON_INSTALL_PREFIX}/ert_gui/models/mixins "${PYTHON_SOURCES}" True)

add_subdirectory(connectorless)
