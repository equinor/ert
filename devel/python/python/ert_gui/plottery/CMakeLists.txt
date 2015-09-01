set(PYTHON_SOURCES
    __init__.py
    ensemble_plot.py
    gen_kw_plot.py
    observation_plot.py
    plot_config.py
    plot_context.py
    plot_tools.py
    probability_plot.py
    summary_plot.py
)

add_python_package("python.ert_gui.plottery" ${PYTHON_INSTALL_PREFIX}/ert_gui/plottery "${PYTHON_SOURCES}" True)

