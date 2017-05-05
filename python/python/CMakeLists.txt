configure_file(test_env.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/test_env.py )

add_subdirectory( res )
if (ERT_BUILD_GUI)
   add_subdirectory( ert_gui )
endif()
add_subdirectory( bin )
