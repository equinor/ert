set(PYTHON_SOURCES
    __init__.py
    observations.py
)

add_python_package("python.ert_gui.models.connectors.plot"  ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors/plot "${PYTHON_SOURCES}" True)

