set(PYTHON_SOURCES
    __init__.py
    activerealizationsmodel.py
    analysismodulevariablesmodel.py
    ertmodel.py
    ertsummary.py
    selectable_list_model.py
    targetcasemodel.py
    valuemodel.py
)

add_python_package("python.ert_gui.ertwidgets.models" ${PYTHON_INSTALL_PREFIX}/ert_gui/ertwidgets/models "${PYTHON_SOURCES}" True)

