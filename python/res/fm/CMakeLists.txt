set(PYTHON_SOURCES
    __init__.py
)



# install_example() is a small utility function which is used to install an
# example file. If a file with the same name already exists in the installation
# directory the example file will not be reinstalled on top.
#
# The decision whether to install the example file or not is evaluated at
# configure - time; i.e. there is a race open between configure time and
# installation. Should ideally use the INSTALL(CODE ...) functionality to defer
# the decision to installation time.

function(install_example src_file destination)

  set (_full_destination "${CMAKE_INSTALL_PREFIX}/${destination}")
  if (EXISTS "${_full_destination}/${src_file}")
    message(STATUS "File: ${_full_destination}/${src_file} already exists - will *not* updated")
  else()
    message(STATUS "An example ${src_file} will be installed ${_full_destination}")
    install(FILES ${src_file} DESTINATION "${_full_destination}")
  endif()

endfunction()

add_subdirectory(ecl)
add_subdirectory(rms)
add_subdirectory(shell)
add_python_package("python.res.fm" ${PYTHON_INSTALL_PREFIX}/res/fm "${PYTHON_SOURCES}" True)

