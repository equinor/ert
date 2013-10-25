set(PYTHON_SOURCES
    __init__.py
    configuration_line_builder.py
    configuration_line_parser.py
    eclipse_keywords.py
    ensemble_keywords.py
    ert_keywords.py
    queue_system_keywords.py
    simulation_control_keywords.py
)

add_python_package("python.ert_gui.ide.keywords"  ${PYTHON_INSTALL_PREFIX}/ert_gui/ide/keywords "${PYTHON_SOURCES}" True)

add_subdirectory(data)
add_subdirectory(definitions)
