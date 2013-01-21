cmake_minimum_required( VERSION 2.6 )
project( ERT C CXX )
include( CheckFunctionExists )
ENABLE_TESTING()

option( BUILD_ERT      "Build the full ERT application - Linux only" OFF)
option( BUILD_ENS_PLOT "Build small Eclipse plotting program - no" OFF)


if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
   set(ERT_LINUX TRUE )
   add_definitions( -DERT_LINUX )
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   set(ERT_WINDOWS TRUE)
   add_definitions( -DERT_WINDOWS )
endif()    


set( CMAKE_C_FLAGS    "-g -O2 -Wall -std=gnu99 -fno-leading-underscore" )
set( CMAKE_CXX_FLAGS  "-g -O2 -Wall" )

include(CMake/ert_check.cmake)
include(CMake/ert_find.cmake)

set( INSTALL_GROUP          ""     CACHE STRING "Group to install as - blank to install as current group")
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)


if (MSVC)
   set(SHARED_LIB OFF)
else()
   option( SHARED_LIB "Build shared libraries" OFF)
endif()

if (SHARED_LIB)
   set( LIBRARY_TYPE SHARED )
else()
   set( LIBRARY_TYPE STATIC )
endif()     



if (MSVC)
   add_definitions( -D__func__="\\"????\\"")
endif()


if (ERT_LINUX)
   set( NEED_LIBM TRUE )
   set( LINK_STATIC FALSE )
   add_definitions( -DHAVE_PROC )
else()
   set( NEED_LIBM FALSE )
   set( LINK_STATIC TRUE )
endif()


configure_file( ${PROJECT_SOURCE_DIR}/CMake/config/ert_build_config.h.in
                ${PROJECT_BINARY_DIR}/ert_build_config.h )

include_directories( ${PROJECT_SOURCE_DIR}/libert_util/include )
include_directories( ${PROJECT_BINARY_DIR}/libert_util/include )
add_subdirectory( libert_util )

include_directories( ${PROJECT_SOURCE_DIR}/libgeometry/include )
add_subdirectory( libgeometry )

if (BUILD_ERT OR BUILD_ENS_PLOT)
   include_directories( ${PROJECT_SOURCE_DIR}/libplot/include )
   add_subdirectory( libplot )
endif()

include_directories( ${PROJECT_SOURCE_DIR}/libecl/include )
add_subdirectory( libecl )

include_directories( ${PROJECT_SOURCE_DIR}/libecl_well/include )
add_subdirectory( libecl_well )


#-----------------------------------------------------------------
if (BUILD_ERT)
#-----------------------------------------------------------------
   try_compile( DLOPEN ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_dlopen.c )
   if (DLOPEN)
      set(NEED_LIBDL OFF)    
   else()
      set(NEED_LIBDL ON)
   endif()

   option(USE_LSF "Include support for LSF" ON)

   include_directories( ${PROJECT_SOURCE_DIR}/libconfig/include )
   add_subdirectory( libconfig )

   include_directories( ${PROJECT_SOURCE_DIR}/libsched/include )
   add_subdirectory( libsched )

   include_directories( ${PROJECT_SOURCE_DIR}/librms/include )
   add_subdirectory( librms )

   include_directories( ${PROJECT_SOURCE_DIR}/libanalysis/include )
   add_subdirectory( libanalysis )

   include_directories( ${PROJECT_SOURCE_DIR}/libjob_queue/include )
   add_subdirectory( libjob_queue )

   include_directories( ${PROJECT_SOURCE_DIR}/libenkf/include )
   add_subdirectory( libenkf )
endif()

