set(PYTHON_SOURCES
    __init__.py
)
add_python_package("python.res"  ${PYTHON_INSTALL_PREFIX}/res "${PYTHON_SOURCES}" True)

set( ERT_SITE_INIT_TARGET  ""  CACHE STRING "Name of optional external site init module for top level res package")
if (EXISTS ${ERT_SITE_INIT_TARGET})
   message(STATUS "Using res site init module: ${ERT_SITE_INIT_TARGET}")
   install(FILES ${ERT_SITE_INIT_TARGET} DESTINATION ${PYTHON_INSTALL_PREFIX}/res RENAME res_site_init.py)
endif()
add_subdirectory(util)
add_subdirectory(analysis)
add_subdirectory(config)
add_subdirectory(enkf)
add_subdirectory(job_queue)
add_subdirectory(sched)
add_subdirectory(test)
add_subdirectory(simulator)
add_subdirectory(fm)

configure_file(res_lib_info_build.py.in   ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res/__res_lib_info.py )
configure_file(res_lib_info_install.py.in ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res_lib_info_install.py )
install(FILES ${PROJECT_BINARY_DIR}/${PYTHON_INSTALL_PREFIX}/res_lib_info_install.py DESTINATION ${PYTHON_INSTALL_PREFIX}/res RENAME __res_lib_info.py)
