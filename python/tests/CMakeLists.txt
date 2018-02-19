set(TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INSTALL_DIRECTORY ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX})

configure_file(
   __init__.py 
   "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/tests/__init__.py"
   @ONLY
)

add_subdirectory(global)

if (GUI)
   add_subdirectory(gui)
endif()


configure_file( test_install.in ${EXECUTABLE_OUTPUT_PATH}/test_install @ONLY )
