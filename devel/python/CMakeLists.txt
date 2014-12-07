set(PYTHON_INSTALL_PREFIX  "python"  CACHE STRING "Subdirectory to install Python modules in")

if (BUILD_TESTS)
   add_subdirectory( test )
endif()
add_subdirectory( python )
