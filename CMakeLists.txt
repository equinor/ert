cmake_minimum_required( VERSION 2.8 )
project( ERT C CXX )

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 OLD)
endif()

#-----------------------------------------------------------------

set( ERT_VERSION_MAJOR 2 )   # Remember to update release notes whenever
set( ERT_VERSION_MINOR 2 )   # you change the ERT_VERSION_MINOR or MAJOR
set( ERT_VERSION_MICRO git ) # with "new in Ert Version X.X.X"!

# If the micro version is not integer, that should be interpreted as a
# development version leading towards version MAJOR.MINOR.0

#-----------------------------------------------------------------

option( BUILD_TESTS         "Should the tests be built"                               OFF)
option( RST_DOC             "Build RST documentation"                                 OFF)
option( USE_RUNPATH         "Should we embed path to libraries"                       ON)
option( GUI                 "Build Python GUI"                                        ON)

find_package( ecl REQUIRED )
find_package( res REQUIRED )

function( make_symlink target link )
  if (EXISTS ${link}) 
     EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E remove "${link}")
  endif()

  if (EXISTS ${target})
     EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E create_symlink "${target}" "${link}")
     message(STATUS "Linking : ${link} -> ${target}")
  else()
     message( SEND_ERROR "Link target: ${target} does not exist")
  endif()    
endfunction()  

set(STATOIL_TESTDATA_ROOT "" CACHE PATH  "Root to Statoil internal testdata")
if (EXISTS ${STATOIL_TESTDATA_ROOT})
  make_symlink( "${STATOIL_TESTDATA_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/test-data/Statoil")
endif()

EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/bin")
make_symlink( "${CMAKE_CURRENT_SOURCE_DIR}/share" "${CMAKE_CURRENT_BINARY_DIR}/share" )
make_symlink( "${res_DIR}/../../bin/job_dispatch.py" "${PROJECT_BINARY_DIR}/bin/job_dispatch.py")

set( ERT_ROOT "${PROJECT_BINARY_DIR}" )


include( CheckFunctionExists )
include( CheckTypeSize )
ENABLE_TESTING()

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
   message(STATUS "Found Linux")
   set(ERT_LINUX TRUE )
   add_definitions( -DERT_LINUX )
   set( ERT_BINARY_POSTFIX .${ERT_VERSION_MAJOR}.${ERT_VERSION_MINOR} )
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   message(STATUS "Found Darwin")
   set(ERT_LINUX TRUE )
   set(ERT_MAC TRUE)
   add_definitions( -DERT_LINUX )
   set( ERT_BINARY_POSTFIX .${ERT_VERSION_MAJOR}.${ERT_VERSION_MINOR} )
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   message(STATUS "Found Windows")
   set(ERT_WINDOWS TRUE)
   add_definitions( -DERT_WINDOWS )
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
   message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
   set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING 
       "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
   # Set the possible values of build type for cmake-gui
   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                "MinSizeRel" "RelWithDebInfo")
endif()

# Treat warnings as errors if not on Windows
if (NOT ERT_WINDOWS)
   set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -std=gnu99 -Wall -Wno-unknown-pragmas ")
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall " )
endif()

if (MSVC)
    add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4996" )
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(CXX11Features)



#-----------------------------------------------------------------

# Checking based on compiling. Some of the code generates warnings, so we just cut down to bare-bone compiler flags.

set( CMAKE_C_FLAGS_main ${CMAKE_C_FLAGS} )
set( CMAKE_CXX_FLAGS_main ${CMAKE_CXX_FLAGS} )

if (NOT ERT_WINDOWS)
  set( CMAKE_C_FLAGS_main "${CMAKE_C_FLAGS} -std=gnu99" )
endif()

set( ERT_EXTERNAL_UTIL_LIBS "" )
include(cmake/ert_build_check.cmake)
include(cmake/ert_api_check.cmake)
include(cmake/ert_lib_check.cmake)

set( CMAKE_C_FLAGS ${CMAKE_C_FLAGS_main} )
set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_main} )

include(cmake/Modules/UseMultiArch.cmake)
include(cmake/ert_link.cmake)

#-----------------------------------------------------------------

set(INSTALL_GROUP          ""     CACHE STRING "Group to install as - blank to install as current group")
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)


set( NEED_LIBM TRUE )
set( LINK_STATIC FALSE )
add_definitions( -DHAVE_PROC )

if (BUILD_TESTS)
   option( ERT_LSF_SUBMIT_TEST "Build and run tests of LSF submit" OFF)
endif()


include(cmake/ert_module_name.cmake)

add_subdirectory( libenkf )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/share DESTINATION ${CMAKE_INSTALL_PREFIX})


add_subdirectory( python )
add_subdirectory( bin )

if(RST_DOC)
   add_subdirectory( docs )
endif()

