cmake_minimum_required( VERSION 2.6 )
project( ERT C )
add_definitions( -g -O2 -Wall -std=gnu99 -m64 )


option(HAVE_ZLIB "Include support for zlib functions compress()/uncompress()" OFF)
find_library( ZLIB_LIBRARY NAMES z )
find_path( ZLIB_HEADER zlib.h /usr/include )

option(HAVE_PTHREAD "Include support for pthreads" OFF)
find_library( PTHREAD_LIBRARY NAMES pthread )
find_path( PTHREAD_HEADER pthread.h /usr/include )

find_path( EXECINFO_HEADER execinfo.h /usr/include )
if (EXECINFO_HEADER)
   set( HAVE_EXECINFO ON )   
endif()

option(HAVE_LSF "Include support for LSF" OFF)
if (HAVE_LSF)
   set( HAVE_LSF OFF )
   #find_path( LSF_HEADER lsf.h ENV{LSF_INCLUDE_PATH})
   find_path( LSF_HEADER lsf.h /Xprog/LSF/8.0/include/lsf)
   find_library( LSF_LIBRARY NAMES lsf PATHS /prog/LSF/8.0/linux2.6-glibc2.3-x86_64/lib )
   if (LSF_HEADER)
      if (LSF_LIBRARY)
          set( HAVE_LSF ON )
      endif()
   endif()
endif()

try_compile( HAVE_FORK ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_fork.c )
option(HAVE_FORK "Have fork() system call")

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

