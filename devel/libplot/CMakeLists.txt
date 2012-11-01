set( PLOT_INSTALL_PREFIX "plot" CACHE STRING "Prefix for installation of libplot")
set( PLPLOT_INCLUDE_PATH "/project/res/x86_64_RH_4/include/plplot" CACHE FILEPATH "Path to PLPLOT include directory")
set( PLPLOT_LIB_PATH     "/project/res/x86_64_RH_4/lib"            CACHE FILEPATH "Path to PLPLOT library path")

find_library( PLPLOT_LIBRARY NAMES  plplotd  PATHS ${PLPLOT_LIB_PATH} )


find_path( PLPLOT_HEADER  plplot/plplot.h  ${PLPLOT_INCLUDE_PATH} )

include_directories( ${libutil_build_path} )
include_directories( ${libutil_src_path} )
include_directories( ${PLPLOT_INCLUDE_PATH} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src )

add_subdirectory( src )
