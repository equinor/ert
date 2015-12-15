# This file contains various checks which will append to the list
# $ERT_EXTERNAL_UTIL_LIBS which should contain all the external library
# dependencies. Observe that all library dependencies go transitively
# through the ert_util library.

if (ERT_HAVE_LAPACK)
   set(CMAKE_REQUIRED_LIBS LAPACK_LIBRARY)
   try_compile( BLAS0 ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/Tests/test_blas.c )

   set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${LAPACK_LIBRARY})
   if (BLAS0)
      set(BUILD_NEED_BLAS OFF)    
   else()
      set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${BLAS_LIBRARY})
      set( ERT_EXTERNAL_UTIL_LIBS ${ERT_EXTERNAL_UTIL_LIBS} ${BLAS_LIBRARY} ${LAPACK_LIBRARY})
   endif()
endif()


if (ERT_WINDOWS)
   find_library( SHLWAPI_LIBRARY NAMES Shlwapi )
   set( ERT_EXTERNAL_UTIL_LIBS ${ERT_EXTERNAL_UTIL_LIBS} Shlwapi )
endif()

if (UNIX)
   set( CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} m )
   set( ERT_EXTERNAL_UTIL_LIBS ${ERT_EXTERNAL_UTIL_LIBS} m )
endif(UNIX)


if (ERT_HAVE_ZLIB)
   set( ERT_EXTERNAL_UTIL_LIBS ${ERT_EXTERNAL_UTIL_LIBS} ${ZLIB_LIBRARY} )
endif()

if (HAVE_PTHREAD)
   set( ERT_EXTERNAL_UTIL_LIBS ${ERT_EXTERNAL_UTIL_LIBS} ${PTHREAD_LIBRARY} )
endif()

find_library( DL_LIBRARY NAMES dl )
find_path( DLFUNC_HEADER dlfcn.h )
if (DL_LIBRARY AND DLFUNC_HEADER)
    set( CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} dl )
    set( ERT_EXTERNAL_UTIL_LIBS ${ERT_EXTERNAL_UTIL_LIBS} ${DL_LIBRARY} )
endif()