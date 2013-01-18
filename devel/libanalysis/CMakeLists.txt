set( ANALYSIS_INSTALL_PREFIX "" CACHE STRING "Prefix for installation of libanalysis")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_subdirectory( src )
