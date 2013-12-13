set(PYTHON_SOURCES
    __init__.py
    plot_data.py
    plot_data_fetcher.py
)

add_python_package("python.ert_gui.tools.plot.data"  ${PYTHON_INSTALL_PREFIX}/ert_gui/tools/plot/data "${PYTHON_SOURCES}" True)

