set(PYTHON_SOURCES
    __init__.py
    cases.py
    ertshell.py
    gen_kw_keys.py
    plugins.py
    results.py
    shell_function.py
    shell_plot.py
    shell_tools.py
    summary_keys.py
    workflows.py
)

add_python_package("python.ert_gui.shell"  ${PYTHON_INSTALL_PREFIX}/ert_gui/shell "${PYTHON_SOURCES}" True)

