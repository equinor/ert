set( RMS_INSTALL_PREFIX "" CACHE STRING "Prefix for installation of librms")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories( ${libecl_src_path}/../include )

add_subdirectory( src )
add_subdirectory( applications )
