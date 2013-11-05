set(PYTHON_SOURCES
    __init__.py
    plot_panel.py
    plot_tool.py
    plot_window.py
)

add_python_package("python.ert_gui.tools.plot"  ${PYTHON_INSTALL_PREFIX}/ert_gui/tools/plot "${PYTHON_SOURCES}" True)

