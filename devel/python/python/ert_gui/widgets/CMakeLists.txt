set(PYTHON_SOURCES
    __init__.py
    error_popup.py
    helped_widget.py
    path_chooser.py
)

add_python_package("python.ert_gui.widgets" ${PYTHON_INSTALL_PREFIX}/ert_gui/widgets "${PYTHON_SOURCES}" True)

