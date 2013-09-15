set( HAVE_LSF_LIBRARY ON )
set( LSF_INCLUDE_PATH $ENV{LSF_INCLUDE_PATH}      CACHE FILEPATH "Path to search for LSF header file lsf/lsf.h")
set( LSF_LIB_PATH     $ENV{LSF_HOME}/lib          CACHE FILEPATH "Path to search for LSF library files")

find_path( LSF_HEADER lsf/lsf.h ${LSF_INCLUDE_PATH})
find_library( LSF_LIBRARY NAMES lsf PATHS ${LSF_LIB_PATH}) 

if (NOT DEFINED ${LSF_HEADER})
  set( HAVE_LSF_LIBRARY OFF)
endif()

if (NOT DEFINED ${LSF_LIBRARY})
  set( HAVE_LSF_LIBRARY OFF )
endif()

if (HAVE_LSF_LIBRARY) 
   include_directories( ${LSF_INCLUDE_PATH} )     
   add_definitions( -DHAVE_LSF_LIBRARY )
endif()

add_subdirectory( src )
if (BUILD_APPLICATIONS)
   add_subdirectory( applications )
endif()

if (BUILD_TESTS)
   add_subdirectory( tests )
endif()


