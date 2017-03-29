add_subdirectory( src )
add_subdirectory( applications/ert_tui )

if (NOT ERT_HAVE_ZLIB)
    message(SEND_ERROR " Package zlib-devel not installed." )
endif()

if (BUILD_TESTS)
   add_subdirectory( tests )
endif()
