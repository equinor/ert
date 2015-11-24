set(TEST_SOURCES
    __init__.py
    test_plot_style.py
)

add_python_package("python.tests.gui.plottery" ${PYTHON_INSTALL_PREFIX}/tests/gui/plottery "${TEST_SOURCES}" False)

addPythonTest(gui.plottery.plot_style gui.plottery.test_plot_style.PlotStyleTest)