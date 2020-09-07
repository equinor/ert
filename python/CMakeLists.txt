if (NOT ENABLE_PYTHON)
    return ()
endif()

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include(init_python)
init_python()

add_subdirectory( res )
add_subdirectory( job_runner )
add_subdirectory( ert_logger )
