add_executable( well_state_test.x well_state_test.c )
add_executable( well_info_test.x well_info_test.c )


set(program_list well_state_test.x well_info_test.x )
foreach(prog ${program_list})
   target_link_libraries( ${prog} well_static ecl_static util_static )
   #-----------------------------------------------------------------
   if (ECL_INSTALL_PREFIX)
      set (destination ${INSTALL_ROOT}/${WELL_INSTALL_PREFIX}/bin)
   else()
      set (destination ${INSTALL_ROOT}/bin)
   endif()

   install(TARGETS ${prog} DESTINATION ${destination})
   if (INSTALL_GROUP)
      install(CODE "EXECUTE_PROCESS(COMMAND chgrp ${INSTALL_GROUP} ${destination}/${prog})")
      install(CODE "EXECUTE_PROCESS(COMMAND chmod g+w ${destination}/${prog})")
   endif()
endforeach()
