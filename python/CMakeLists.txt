if (NOT ENABLE_PYTHON)
    return ()
endif()

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include(init_python)
init_python(2.7)

#-----------------------------------------------------------------
# Find an existing ert.ecl disribution

find_python_package(ecl 2.1.0 ${PYTHON_INSTALL_PREFIX})
find_python_package(cwrap 0.0.0 ${PYTHON_INSTALL_PREFIX})
find_python_package(numpy 1.7.1 ${PYTHON_INSTALL_PREFIX})
find_python_package(yaml  3.1.0 ${PYTHON_INSTALL_PREFIX})

# libres python needs ecl, cwrap and numpy
if (NOT DEFINED PY_ecl OR NOT DEFINED PY_cwrap OR NOT DEFINED PY_numpy)
    message(WARNING "ecl, cwrap, or numpy module not found - Python wrappers not enabled")
    return()
endif()

add_subdirectory( res )
add_subdirectory( job_runner )
