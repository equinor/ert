set(TEST_SOURCES
    __init__.py
    test_plot_data.py
    test_plot_settings.py
)

add_python_package("python.tests.res.enkf.plot" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/plot "${TEST_SOURCES}" False)

python_config_test(tests.res.enkf.plot.test_plot_settings.PlotSettingsTest)
python_config_test(tests.res.enkf.plot.test_plot_data.PlotDataTest)
