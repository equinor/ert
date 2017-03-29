include(cmake/ert_module.cmake)

add_subdirectory( script )
add_subdirectory( src )
add_subdirectory( modules )

if (NOT ERT_HAVE_LAPACK)
    message (SEND_ERROR " Package Lapack not installed" )
endif()

if (BUILD_TESTS)
   add_subdirectory( tests )
endif()

if (BUILD_APPLICATIONS)
   add_subdirectory( applications )
endif()

