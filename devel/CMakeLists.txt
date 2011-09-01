cmake_minimum_required( VERSION 2.6 )

include (CheckFunctionExists)
project( ERT C )
add_definitions( -g -O2 -Wall -std=gnu99 )

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

find_library( LAPACK_LIBRARY NAMES lapack PATHS C:/lib)
find_library( BLAS_LIBRARY NAME blas PATHS C:/lib)
find_library( GFORTRAN_LIBRARY NAME gfortran-3 )

#-----------------------------------------------------------------
find_library( SHLWAPI_LIBRARY NAMES Shlwapi )
if (SHLWAPI_LIBRARY)
   message("Found the SHLWAPI library")
endif()

check_function_exists( realpath HAVE_RELPATH )
if (HAVE_REALPATH)
   add_definitions( -DHAVE_REALPATH )
endif()

check_function_exists( fork HAVE_FORK )
if (HAVE_FORK)
   add_definitions( -DHAVE_FORK )
endif()

check_function_exists( symlink HAVE_SYMLINK )
if (HAVE_SYMLINK)
   add_definitions( -DHAVE_SYMLINK )
endif()

check_function_exists( getuid HAVE_GETUID )
if (HAVE_GETUID)
   add_definitions( -DHAVE_GETUID )
endif()

check_function_exists( localtime_r HAVE_LOCALTIME_R )
if (HAVE_LOCALTIME_R)
   add_definitions( -DHAVE_LOCALTIME_R )
endif()

try_compile( MKDIR_POSIX ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_mkdir.c )
if (MKDIR_POSIX)
   add_definitions( -DMKDIR_POSIX )
endif()

check_function_exists( lockf HAVE_LOCKF )
if (HAVE_LOCKF)
   add_definitions( -DHAVE_LOCKF )
endif()

check_function_exists( glob HAVE_GLOB )
if (HAVE_GLOB)
   add_definitions( -DHAVE_GLOB )
endif()

check_function_exists( fnmatch HAVE_FNMATCH )
if (HAVE_FNMATCH)
   add_definitions( -DHAVE_FNMATCH )
endif()

check_function_exists( fsync HAVE_FSYNC )
if (HAVE_FSYNC)
   add_definitions( -DHAVE_FSYNC )
endif()

#try_compile( HAVE_GETENV ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_getenv.c )
check_function_exists( getenv HAVE_GETENV )
if (HAVE_GETENV)
   add_definitions( -DHAVE_GETENV )
endif()


set(CMAKE_REQUIRED_FLAGS _GNU_SOURCE __USE_GNU)
check_function_exists( pow10 HAVE_POW10 )
if (HAVE_POW10)
   add_definitions( -DHAVE_POW10 )
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

