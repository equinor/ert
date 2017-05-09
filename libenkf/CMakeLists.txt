add_subdirectory( src )

#set( CMAKE_REQUIRED_LIBRARIES ${UTIL_LIBRARY} )
#check_function_exists( buffer_fwrite_compressed HAVE_BUFFER_COMPRESS )
#if (NOT HAVE_BUFFER_COMPRESS)
#    message( SEND_ERROR "missing zlib support in buffer")
#endif()

if (BUILD_TESTS)
   add_subdirectory( tests )
endif()
