configure_file(test_env.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/test_env.py )
add_subdirectory( bin )
add_subdirectory( tiny_ert_dash )
if (GUI)
   add_subdirectory( ert_gui )
endif()


