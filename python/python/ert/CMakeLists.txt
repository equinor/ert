# -----------------------------------------------------------------
# Here is two blocks of code linking python packages and shared
# libraries respectively from the buid directory to the location of
# the libecl library, this is to ensure that the resulting build
# directory can be used stand alone.

EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/ert")
set(PACKAGE_LIST test util ecl geo)
foreach (package ${PACKAGE_LIST})
   set(LINK ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/ert/${package} )
   set(TARGET ${ERT_PYTHONPATH}/ert/${package} )
   message(STATUS "Linking Python package ${package}: ${LINK} -> ${TARGET}")
   EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E create_symlink "${TARGET}" "${LINK}")
endforeach()

EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
foreach (lib ${UTIL_LIBRARY} ${ECL_LIBRARY} ${GEO_LIBRARY})
   file(GLOB lib_list "${lib}*")
   foreach( lib_file ${lib_list} )
       get_filename_component( name ${lib_file} NAME )
       set(LINK ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${name})
       message(STATUS "Linking library: ${LINK} -> ${lib_file}")
       EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E create_symlink "${lib_file}" "${LINK}")
   endforeach()    
endforeach()

# -----------------------------------------------------------------

set(PYTHON_SOURCES
    __init__.py
)
add_python_package("python.ert"  ${PYTHON_INSTALL_PREFIX}/ert "${PYTHON_SOURCES}" True)

IF (BUILD_ERT)
   set( ERT_SITE_INIT_TARGET  ""  CACHE FILE "Name of optional external site init module for top level ert package")
   if (EXISTS ${ERT_SITE_INIT_TARGET})
      message(STATUS "Using ert site init module: ${ERT_SITE_INIT_TARGET}")
      install(FILES ${ERT_SITE_INIT_TARGET} DESTINATION ${PYTHON_INSTALL_PREFIX}/ert RENAME ert_site_init.py)
   endif()
   add_subdirectory(analysis)
   add_subdirectory(config)
   add_subdirectory(enkf)
   add_subdirectory(job_queue)
   add_subdirectory(rms)
   add_subdirectory(sched)
   add_subdirectory(server)
ENDIF()

configure_file(res_lib_info_build.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/ert/__res_lib_info.py )
configure_file(res_lib_info_install.py.in ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res_lib_info_install.py )
install(FILES ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res_lib_info_install.py DESTINATION ${PYTHON_INSTALL_PREFIX}/ert RENAME __res_lib_info.py)
