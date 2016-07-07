set(PYTHON_SOURCES
    __init__.py
    add_remove_widget.py
    clearable_line_edit.py
    custom_date_edit.py
    error_popup.py
    helped_widget.py
    legend.py
    path_chooser.py
    path_format.py
    progress.py
    search_box.py
    simple_progress.py
    string_box.py
    util.py
)

add_python_package("python.ert_gui.widgets" ${PYTHON_INSTALL_PREFIX}/ert_gui/widgets "${PYTHON_SOURCES}" True)

