set(PYTHON_SOURCES
    __init__.py
    active_realizations_model.py
    ertmodel.py
    ertsummary.py
    selectable_list_model.py
    value_model.py
)

add_python_package("python.ert_gui.ertwidgets.models" ${PYTHON_INSTALL_PREFIX}/ert_gui/ertwidgets/models "${PYTHON_SOURCES}" True)

