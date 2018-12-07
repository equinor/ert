find_program(FLOW_EXECUTABLE
  NAMES "flow"
  DOC "Flow reservoir simulator")


if(FLOW_EXECUTABLE)
  execute_process(COMMAND ${FLOW_EXECUTABLE} --version
                  OUTPUT_VARIABLE flow_version
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (flow_version MATCHES "^flow [0-9]")
    string(REPLACE "flow " "" FLOW_VERSION "${flow_version}")
  else()
    # Eearly versions of flow do not respond to "flow --version", we therefor
    # just guess that the version we have found is the last version before "flow
    # --version" worked - i.e. 2018.10.
    message(STATUS "\"flow --version\" did not return version information - assuming 2018.10")
    set(FLOW_VERSION "2018.10")
  endif()
  unset(flow_version)
endif()

find_package_handle_standard_args(Flow
                                  REQUIRED_VARS FLOW_EXECUTABLE
                                  VERSION_VAR FLOW_VERSION)


mark_as_advanced(FLOW_EXECUTABLE)
mark_as_advanced(FLOW_VERSION)
