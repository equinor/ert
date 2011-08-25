cmake_minimum_required( VERSION 2.6 )
project( ERT C )
add_definitions( -g -O2 -Wall -std=gnu99 -m64 )

#-----------------------------------------------------------------
option(USE_ZLIB "Include support for zlib functions compress()/uncompress()" ON)
if (USE_ZLIB)
   find_library( ZLIB_LIBRARY NAMES z )
   find_path( ZLIB_HEADER zlib.h /usr/include )

   if (!ZLIB_LIBRARY)
      set( USE_ZLIB OFF )
   endif()

   if (!ZLIB_HEADER)
      set( USE_ZLIB OFF )
   endif()
endif()
if (USE_ZLIB)
   set( HAVE_ZLIB ON )
   add_definitions( -DHAVE_ZLIB )
endif()
#-----------------------------------------------------------------
option( USE_PTHREAD "Include support for pthreads" ON )
if (USE_PTHREAD)
   find_library( PTHREAD_LIBRARY NAMES pthread )
   find_path( PTHREAD_HEADER pthread.h /usr/include )

   if (!PTHREAD_LIBRARY)
      set( USE_PTHREAD OFF )
   endif()

   if (!PTHREAD_HEADER)
      set( USE_PTHREAD OFF )
   endif()
endif()
if (USE_PTHREAD)
   set( HAVE_PTHREAD ON )
   add_definitions( -DHAVE_PTHREAD )
endif()
#-----------------------------------------------------------------
option(USE_LSF "Include support for LSF" ON)
if (USE_LSF)
   #find_path( LSF_HEADER lsf.h ENV{LSF_INCLUDE_PATH})
   find_path( LSF_HEADER lsf.h /prog/LSF/8.0/include/lsf)
   find_library( LSF_LIBRARY NAMES lsf PATHS /prog/LSF/8.0/linux2.6-glibc2.3-x86_64/lib )   
   if (!LSF_HEADER)
      set(USE_LSF OFF)
   endif()
   if (!LSF_LIBRARY)
      set( USE_LSF OFF )
   endif()
endif()
if (USE_LSF)
   set( HAVE_LSF ON )
   add_definitions( -DHAVE_LSF )
endif()
#-----------------------------------------------------------------
find_path( EXECINFO_HEADER execinfo.h /usr/include )
if (EXECINFO_HEADER)
   set( HAVE_EXECINFO ON )   
   add_definitions( -DHAVE_EXECINFO )
endif()
#-----------------------------------------------------------------

#find_library( LAPACK_LIBRARY NAMES lapack )

#-----------------------------------------------------------------


try_compile( HAVE_REALPATH ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_realpath.c )
if (HAVE_REALPATH)
   add_definitions( -DHAVE_REALPATH )
endif()

try_compile( HAVE_FORK ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_fork.c )
if (HAVE_FORK)
   add_definitions( -DHAVE_FORK )
endif()

try_compile( HAVE_SYMLINK ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_symlink.c )
if (HAVE_SYMLINK)
   add_definitions( -DHAVE_SYMLINK )
endif()

try_compile( HAVE_GETUID ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_getuid.c )
if (HAVE_GETUID)
   add_definitions( -DHAVE_GETUID )
endif()


configure_file( ${PROJECT_SOURCE_DIR}/CMake/config/ert_build_config.h.in
                ${PROJECT_BINARY_DIR}/ert_build_config.h )

set( libutil_build_path ${CMAKE_BINARY_DIR}/libutil/src )
set( libutil_src_path   ${CMAKE_SOURCE_DIR}/libutil/src )
set( libconfig_src_path ${CMAKE_SOURCE_DIR}/libconfig/src )
set( libecl_src_path    ${CMAKE_SOURCE_DIR}/libecl/src )


add_subdirectory( libutil )
add_subdirectory( libecl )
#add_subdirectory( libconfig )
#add_subdirectory( libjob_queue )


#add_subdirectory( ${LIBSCHED_PATH} )
#add_subdirectory( ${LIBECL_PATH} )
#add_subdirectory( ${LIBCONFIG_PATH} )

#add_subdirectory( ${LIBPLOT_PATH} )
#add_subdirectory( ${LIBENKF_PATH} )

