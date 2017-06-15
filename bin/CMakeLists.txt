set( ERT_ROOT ${PROJECT_BINARY_DIR})
configure_file(ert.in  ${EXECUTABLE_OUTPUT_PATH}/ert )
configure_file(gui_test.in ${EXECUTABLE_OUTPUT_PATH}/gui_test)

set( ERT_ROOT ${CMAKE_INSTALL_PREFIX})
configure_file(ert.in  ${EXECUTABLE_OUTPUT_PATH}/ert_install )


install(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/ert_install DESTINATION ${CMAKE_INSTALL_PREFIX}/bin RENAME ert)
