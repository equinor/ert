set(PYTHON_SOURCES
    __init__.py
    data_file_handler.py
    define_handler.py
    grid_handler.py
    init_section_handler.py
    install_job_handler.py
    keyword_handler.py
    num_realizations_handler.py
    queue_option_handler.py
    queue_system_handler.py
    schedule_file_handler.py
    unknown_handler.py
)

add_python_package("python.ert_gui.ide.handlers"  ${PYTHON_INSTALL_PREFIX}/ert_gui/ide/handlers "${PYTHON_SOURCES}" True)

