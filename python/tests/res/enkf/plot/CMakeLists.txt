set(TEST_SOURCES
    __init__.py
    test_plot_data.py
    test_plot_settings.py
)

add_python_package("python.tests.res.enkf.plot" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/plot "${TEST_SOURCES}" False)

addPythonTest(res.enkf.plot.plot_settings tests.res.enkf.plot.test_plot_settings.PlotSettingsTest)
if (STATOIL_TESTDATA_ROOT)
  addPythonTest(res.enkf.plot.plot_block_data tests.res.enkf.plot.test_plot_data.PlotDataTest LABELS StatoilData)
endif()
