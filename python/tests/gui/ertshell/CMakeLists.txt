set(TEST_SOURCES
    __init__.py
    ert_shell_test_context.py
    test_ertshell_analysis.py
    test_ertshell_context.py
    test_ertshell_observations.py
    test_ertshell_plot_settings.py
    test_ertshell_smoother.py
    test_ertshell_server.py

    test_shell_function.py
    test_shell_property.py
    test_shell_collection.py
)

add_python_package("python.tests.gui.ertshell"  ${PYTHON_INSTALL_PREFIX}/tests/gui/ertshell "${TEST_SOURCES}" False)

addPythonTest(tests.gui.ertshell.test_ertshell_analysis.ErtShellAnalysisModuleTest)
addPythonTest(tests.gui.ertshell.test_ertshell_context.ErtShellContextTest)
addPythonTest(tests.gui.ertshell.test_ertshell_observations.ErtShellObservationsTest)
addPythonTest(tests.gui.ertshell.test_ertshell_plot_settings.ErtShellPlotSettingsTest)
addPythonTest(tests.gui.ertshell.test_ertshell_smoother.ErtShellSmootherTest)
addPythonTest(tests.gui.ertshell.test_ertshell_server.ErtShellServerTest)
addPythonTest(tests.gui.ertshell.test_shell_function.ShellFunctionTest)
addPythonTest(tests.gui.ertshell.test_shell_property.ShellPropertyTest)
addPythonTest(tests.gui.ertshell.test_shell_collection.ShellCollectionTest)
