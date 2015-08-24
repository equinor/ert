set(PYTHON_SOURCES
    __init__.py
    ensemble_plot.py
    plot_config.py
    plot_context.py
    summary_plot.py
)

add_python_package("python.ert_gui.plottery" ${PYTHON_INSTALL_PREFIX}/ert_gui/plottery "${PYTHON_SOURCES}" True)

