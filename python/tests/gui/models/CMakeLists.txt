set(TEST_SOURCES
    __init__.py
    test_base_run_model.py
)

add_python_package("python.tests.gui.models" ${PYTHON_INSTALL_PREFIX}/tests/gui/models "${TEST_SOURCES}" False)

addPythonTest(tests.gui.models.test_base_run_model.BaseRunModelTest)
