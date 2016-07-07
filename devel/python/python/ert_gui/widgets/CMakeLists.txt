set(PYTHON_SOURCES
    __init__.py
    add_remove_widget.py
    checkbox.py
    clearable_line_edit.py
    closable_dialog.py
    combo_choice.py
    custom_date_edit.py
    custom_dialog.py
    double_box.py
    double_spinner.py
    error_popup.py
    helped_widget.py
    legend.py
    list_edit_box.py
    path_chooser.py
    path_format.py
    progress.py
    search_box.py
    simple_progress.py
    string_box.py
    util.py
    validated_dialog.py
    workflow_dialog.py
)

add_python_package("python.ert_gui.widgets" ${PYTHON_INSTALL_PREFIX}/ert_gui/widgets "${PYTHON_SOURCES}" True)

