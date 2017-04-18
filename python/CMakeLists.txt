include(cmake/find_python_module.cmake)
include(cmake/python_package_versions.cmake) # finds version

if (EXISTS "/etc/debian_version")
   set( PYTHON_PACKAGE_PATH "dist-packages")
else()
   set( PYTHON_PACKAGE_PATH "site-packages")
endif()

set( PY_VERSION 2.7 )
FIND_PACKAGE(PythonInterp ${PY_VERSION} EXACT)
if (NOT DEFINED PYTHON_EXECUTABLE)
   message(WARNING "Python${PY_VERSION} interpreter not found - Python wrappers not enabled")
   return()
endif()

set(PYTHON_INSTALL_PREFIX  "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/${PYTHON_PACKAGE_PATH}"  CACHE STRING "Subdirectory to install Python modules in")

#-----------------------------------------------------------------
# Find an existing ert.ecl disribution

foreach( PREFIX_PATH ${CMAKE_PREFIX_PATH} )
   set(tmp_PYTHONPATH "${PREFIX_PATH}/${PYTHON_INSTALL_PREFIX}:$ENV{PYTHONPATH}")
   set(ENV{PYTHONPATH} ${tmp_PYTHONPATH}) 
   python_package(ert 2.1.0)
   if (DEFINED PY_ert)
      break()
   endif()
endforeach()


if (NOT DEFINED PY_ert)
   message(SEND_ERROR "Existing ert installation not found") 
endif()
set( ERT_PYTHONPATH ${PY_ert_PATH} )

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import ert; print ert.ert_lib_path"
                  OUTPUT_VARIABLE ERT_SO_PATH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(LD_PATH ${ERT_SO_PATH}:$ENV{LD_LIBRARY_PATH})
set(ENV{LD_LIBRARY_PATH} ${LD_PATH})
message(STATUS "Setting LD_LIBRARY_PATH: ${LD_PATH}")

#-----------------------------------------------------------------

python_package(numpy 1.7.1)
if (NOT DEFINED PY_numpy)
   message(WARNING "numpy module not found - Python wrappers not enabled")
   return()
endif()

#-----------------------------------------------------------------

if (ERT_BUILD_GUI)
   python_package(PyQt4 4.8.0)
   python_package( matplotlib 1.2.0  )
   python_package( pandas     0.15.1)
   python_package( scipy      0.16.1 )
   if(NOT DEFINED PY_PyQt4 OR NOT DEFINED PY_matplotlib OR NOT DEFINED PY_scipy OR NOT DEFINED PY_pandas)
     message(SEND_ERROR "Cannot build GUI without PyQt4, matplotlib, scipy, and pandas")
   endif()
endif()


if (BUILD_TESTS)
   add_subdirectory( tests )
endif()
add_subdirectory( python )
