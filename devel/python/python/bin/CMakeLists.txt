set (destination ${CMAKE_INSTALL_PREFIX}/bin)

if (INSTALL_ERT)
   install(PROGRAMS gert DESTINATION ${destination})
   if (INSTALL_GROUP)
      install(CODE "EXECUTE_PROCESS(COMMAND chgrp ${INSTALL_GROUP} ${destination}/gert)")
      install(CODE "EXECUTE_PROCESS(COMMAND chmod g+w ${destination}/gert)")
   endif()
endif()

