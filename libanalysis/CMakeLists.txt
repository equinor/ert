include(cmake/ert_module.cmake)

add_subdirectory( script )
add_subdirectory( src )
add_subdirectory( modules )

set( CMAKE_REQUIRED_LIBRARIES ${UTIL_LIBRARY} )
check_function_exists( matrix_det HAVE_MATRIX_LAPACK )
if (NOT HAVE_MATRIX_LAPACK)
    message( SEND_ERROR "missing linear algebra support in matrix")
endif()

if (BUILD_TESTS)
   add_subdirectory( tests )
endif()

if (BUILD_APPLICATIONS)
   add_subdirectory( applications )
endif()

