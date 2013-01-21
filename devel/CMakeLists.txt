cmake_minimum_required( VERSION 2.6 )
project( ERT C CXX )
include( CheckFunctionExists )
ENABLE_TESTING()


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

set( INSTALL_GROUP          ""     CACHE STRING "Group to install as - blank to install as current group")
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

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

#-----------------------------------------------------------------
find_library( ZLIB_LIBRARY NAMES z )
find_path( ZLIB_HEADER zlib.h /usr/include )

if (ZLIB_LIBRARY AND ZLIB_HEADER)
   option(WITH_ZLIB "Include support for zlib functions compress()/uncompress()" ON)
   if (WITH_ZLIB)
      add_definitions( -DWITH_ZLIB )
   endif()
else()
   set( WITH_ZLIB FALSE )
   message("ZLib not found - zlib support will not be included." )       
endif()
#-----------------------------------------------------------------
find_library( PTHREAD_LIBRARY NAMES pthread )
if (PTHREAD_LIBRARY)
   option( WITH_PTHREAD "Include support for pthreads" ON )
   if (WITH_PTHREAD)
     add_definitions( -DWITH_PTHREAD )
   endif()
else()
   set( WITH_PTHREAD FALSE )
   message("pthread library not found - pthread support will not be included")
endif()
#-----------------------------------------------------------------
find_library( LAPACK_LIBRARY NAMES lapack)
if (LAPACK_LIBRARY)
   set(CMAKE_REQUIRED_LIBS LAPACK_LIBRARY)
   try_compile( BLAS0 ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_blas.c )
   if (BLAS0)
      set(NEED_BLAS OFF)    
   else()
      set(NEED_BLAS ON)
      find_library( BLAS_LIBRARY NAMES blas)
   endif()
   option(WITH_LAPACK "Build LAPACK enabled code" ON)
   if (WITH_LAPACK)
      add_definitions( -DWITH_LAPACK )
   endif()
else()
   set( WITH_LAPACK OFF)     
   message("LAPACK library not found - LAPACK support will not be included")
endif()
#-----------------------------------------------------------------
find_program(LATEX_PATH NAMES pdflatex)
if (LATEX_PATH)
   option( WITH_LATEX "Build small class for compiling LaTeX files" ON)
   if (WITH_LATEX)
      set( WITH_LATEX ON)
      add_definitions( -DWITH_LATEX )
   endif()
else()
   set( WITH_LATEX OFF )
endif()
#-----------------------------------------------------------------
find_path( EXECINFO_HEADER execinfo.h /usr/include )
if (EXECINFO_HEADER)
  add_definitions( -DHAVE_EXECINFO )
endif()
#-----------------------------------------------------------------
find_path( GETOPT_HEADER getopt.h /usr/include )
if (GETOPT_HEADER)
   add_definitions( -DHAVE_GETOPT )
endif()
#-----------------------------------------------------------------
if (ERT_WINDOWS)
   find_library( SHLWAPI_LIBRARY NAMES Shlwapi )
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

set( BUILD_LIBPLOT ON )
if (BUILD_LIBPLOT)
   include_directories( ${PROJECT_SOURCE_DIR}/libplot/include )
   add_subdirectory( libplot )
endif()

include_directories( ${PROJECT_SOURCE_DIR}/libecl/include )
add_subdirectory( libecl )

include_directories( ${PROJECT_SOURCE_DIR}/libecl_well/include )
add_subdirectory( libecl_well )


option( BUILD_ERT      "Build the full ERT application - Linux only" OFF)
option( BUILD_ENS_PLOT "Build small Eclipse plotting program - no" OFF)

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

