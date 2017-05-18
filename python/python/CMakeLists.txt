configure_file(test_env.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/test_env.py )

add_subdirectory( res )

if (INSTALL_ERT_LEGACY)
   add_subdirectory( legacy/ert )
endif()
