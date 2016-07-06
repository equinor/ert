set(PYTHON_SOURCES
    __init__.py
    case_list.py
    case_selector.py
)

add_python_package("python.ert_gui.models.connectors.init"  ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors/init "${PYTHON_SOURCES}" True)

