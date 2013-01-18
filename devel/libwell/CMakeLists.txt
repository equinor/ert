set( WELL_INSTALL_PREFIX "" CACHE STRING "Prefix for installation of libwell")

include_directories( ${libecl_src_path}/../include )

add_subdirectory( src )
add_subdirectory( applications )
