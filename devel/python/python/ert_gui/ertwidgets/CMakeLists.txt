set(PYTHON_SOURCES
    __init__.py
    activelabel.py
    caselist.py
    caseselector.py
    checklist.py
    customdialog.py
    listeditbox.py
    summarypanel.py
    validateddialog.py
    validationsupport.py
)

add_python_package("python.ert_gui.ertwidgets" ${PYTHON_INSTALL_PREFIX}/ert_gui/ertwidgets "${PYTHON_SOURCES}" True)

add_subdirectory(models)
