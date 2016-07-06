set(PYTHON_SOURCES
    __init__.py
    ensemble_resizer.py
    ert_summary.py
)

add_python_package("python.ert_gui.models.connectors"  ${PYTHON_INSTALL_PREFIX}/ert_gui/models/connectors "${PYTHON_SOURCES}" True)

add_subdirectory(run)

