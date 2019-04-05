set(PYTHON_SOURCES
    __init__.py
    about_dialog.py
    ert_splash.py
    ertnotifier.py
    ertplot.py
    gert_main.py
    main_window.py
    newconfig.py
)
configure_file(ert_version.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/ert_gui/ert_version.py )
install(FILES ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/ert_gui/ert_version.py DESTINATION ${PYTHON_INSTALL_PREFIX}/ert_gui)

add_python_package("python.ert_gui" ${PYTHON_INSTALL_PREFIX}/ert_gui "${PYTHON_SOURCES}" True)

add_subdirectory(ertwidgets)
add_subdirectory(ide)
add_subdirectory(plottery)
add_subdirectory(shell)
add_subdirectory(simulation)
add_subdirectory(tools)
