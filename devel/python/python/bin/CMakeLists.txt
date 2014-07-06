set (destination ${CMAKE_INSTALL_PREFIX}/bin)

if (INSTALL_ERT)
   set( application_list )

   foreach(prog ${application_list} )
      install(PROGRAMS ${prog} DESTINATION ${destination})
      if (INSTALL_GROUP)
         install(CODE "EXECUTE_PROCESS(COMMAND chgrp ${INSTALL_GROUP} ${destination}/${prog})")
         install(CODE "EXECUTE_PROCESS(COMMAND chmod g+w ${destination}/${prog})")
      endif()
   endforeach()
endif()

