find_package(Sphinx REQUIRED)
if (SPHINX_FOUND)

  add_custom_target(manual ALL
    COMMAND sphinx-build -d htmldir ${CMAKE_CURRENT_SOURCE_DIR}/manual ${PROJECT_BINARY_DIR}/docs/manual
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

else()
  message(STATUS "Sphinx documentation tool not found - documentation not generated")
endif()
