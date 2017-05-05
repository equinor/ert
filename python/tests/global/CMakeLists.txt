set( TEST_SOURCES
     __init__.py
     test_import.py
     test_import_gui.py
)

add_python_package("python.tests.global"  "${PYTHON_INSTALL_PREFIX}/tests/global" "${TEST_SOURCES}" False)

addPythonTest( tests.global.test_import.ImportRes )


if (ERT_BUILD_GUI)
    addPythonTest( tests.global.test_import_gui.ImportGUI )
endif()
