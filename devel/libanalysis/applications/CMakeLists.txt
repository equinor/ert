add_executable( ert_module_test ert_module_test.c )
target_link_libraries( ert_module_test analysis ert_util )

if (USE_RUNPATH)
   add_runpath( ert_module_test )
endif()

set (destination ${CMAKE_INSTALL_PREFIX}/bin)

install(TARGETS ert_module_test DESTINATION ${destination})

if (INSTALL_GROUP)
   install(CODE "EXECUTE_PROCESS(COMMAND chgrp ${INSTALL_GROUP} ${destination}/ert_module_test)")
   install(CODE "EXECUTE_PROCESS(COMMAND chmod g+w ${destination}/ert_module_test)")
endif()
