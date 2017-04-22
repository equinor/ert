configure_file(test_env.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/test_env.py )

# Add link in build directory to cwrap package
EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}")
set(LINK ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/cwrap )
set(TARGET ${CWRAP_PYTHONPATH}/cwrap )
message(STATUS "Linking Python package ${package}: ${LINK} -> ${TARGET}")
EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E create_symlink "${TARGET}" "${LINK}")


add_subdirectory( ert )
if (BUILD_ERT)
   if (ERT_BUILD_GUI)
      add_subdirectory( ert_gui )
   endif()
endif()
add_subdirectory( bin )
