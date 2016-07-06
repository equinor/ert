set(PYTHON_SOURCES
    __init__.py
    ertmodel.py
    selectable_list_model.py
)

add_python_package("python.ert_gui.ertwidgets.models" ${PYTHON_INSTALL_PREFIX}/ert_gui/ertwidgets/models "${PYTHON_SOURCES}" True)

