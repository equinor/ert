cmake_minimum_required( VERSION 2.6 )
include( CheckFunctionExists )
project( ERT C CXX )

set( CMAKE_C_FLAGS    "-g -O2 -Wall -std=gnu99" )
set( CMAKE_CXX_FLAGS  "-g -O2 -Wall" )


set( INSTALL_GROUP          ""     CACHE STRING "Group to install as - blank to install as current group")


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

  if (NOT PTHREAD_LIBRARY)
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
option(USE_OPENMP "Include OpenMP support" ON)
set(OPENMP USE_OPENMP)
#-----------------------------------------------------------------
option(USE_LSF "Include support for LSF" ON)
#-----------------------------------------------------------------
find_path( EXECINFO_HEADER execinfo.h /usr/include )
if (EXECINFO_HEADER)
  set( HAVE_EXECINFO ON )   
  add_definitions( -DHAVE_EXECINFO )
endif()
#-----------------------------------------------------------------

#-----------------------------------------------------------------
find_library( SHLWAPI_LIBRARY NAMES Shlwapi )

check_function_exists( realpath HAVE_REALPATH )
if (HAVE_REALPATH)
  add_definitions( -DHAVE_REALPATH )
endif()

check_function_exists( fork HAVE_FORK )
if (HAVE_FORK)
  add_definitions( -DHAVE_FORK )
endif()

check_function_exists( readlinkat HAVE_READLINKAT )
if (HAVE_READLINKAT)
   add_definitions( -DHAVE_READLINKAT )
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

check_function_exists( setenv HAVE_SETENV )
if (HAVE_SETENV)
  add_definitions( -DPOSIX_SETENV )
endif()


find_library( LAPACK_LIBRARY NAMES lapack)
set(CMAKE_REQUIRED_LIBS LAPACK_LIBRARY)
try_compile( BLAS0 ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/CMake/Tests/test_blas.c )
if (BLAS0)
   set(NEED_BLAS OFF)    
else()
   set(NEED_BLAS ON)
   find_library( BLAS_LIBRARY NAMES blas)
endif()

configure_file( ${PROJECT_SOURCE_DIR}/CMake/config/ert_build_config.h.in
                ${PROJECT_BINARY_DIR}/ert_build_config.h )

set( libutil_build_path   ${CMAKE_BINARY_DIR}/libutil/src )
set( libutil_src_path     ${CMAKE_SOURCE_DIR}/libutil/src )
set( libecl_src_path      ${CMAKE_SOURCE_DIR}/libecl/src )
set( libgeometry_src_path ${CMAKE_SOURCE_DIR}/libgeometry/src )
set( libwell_src_path     ${CMAKE_SOURCE_DIR}/libwell/src )

set( BUILD_CXX ON )
set( BUILD_ERT OFF)

add_subdirectory( libutil )
add_subdirectory( libgeometry )
add_subdirectory( libecl )
add_subdirectory( libwell )

add_dependencies( geometry_shared util_shared )
add_dependencies( ecl_shared      util_shared geometry_shared )
add_dependencies( well_shared     ecl_shared  util_shared geometry_shared )

add_dependencies( geometry_static util_static )
add_dependencies( ecl_static      util_static geometry_static )
add_dependencies( well_static     ecl_static  util_static geometry_static )



#-----------------------------------------------------------------
if (BUILD_ERT)
   set( libconfig_src_path    ${CMAKE_SOURCE_DIR}/libconfig/src )
   set( libsched_src_path     ${CMAKE_SOURCE_DIR}/libsched/src )
   set( librms_src_path       ${CMAKE_SOURCE_DIR}/librms/src )
   set( libanalysis_src_path  ${CMAKE_SOURCE_DIR}/libanalysis/src )
   set( libjob_queue_src_path ${CMAKE_SOURCE_DIR}/libjob_queue/src )
   set( libplot_src_path      ${CMAKE_SOURCE_DIR}/libplot/src )
   set( libenkf_src_path      ${CMAKE_SOURCE_DIR}/libenkf/src )

   add_subdirectory( libconfig )
   add_subdirectory( libanalysis )
   add_subdirectory( librms )
   add_subdirectory( libsched )
   add_subdirectory( libjob_queue )
   add_subdirectory( libplot )
   add_subdirectory( libenkf)

   add_dependencies( config_shared util_shared )
   #add_dependencies( analysis_shared util_shared )
   add_dependencies( rms_shared geometry_shared util_shared )
   add_dependencies( sched_shared util_shared ecl_shared )
   add_dependencies( job_queue_shared config_shared util )
   add_dependencies( plot_shared util_shared )
   add_dependencies( enkf_shared plot_shared ecl_shared util_shared config_shared sched_shared rms_shared analysis job_queue_shared )
   add_dependencies( ert_shared enkf_shared plot_shared )

   add_dependencies( config_static util_static )
   #add_dependencies( analysis_static util_static )
   add_dependencies( rms_static geometry_static util_static )
   add_dependencies( sched_static util_static ecl_static )
   add_dependencies( job_queue_static config_static util )
   add_dependencies( plot_static util_static )
   add_dependencies( enkf_static plot_static ecl_static util_static config_static sched_static rms_static analysis job_queue_static )
   add_dependencies( ert enkf_static plot_static )
endif()

