cmake_minimum_required(VERSION 2.8.12)
project(res C)

include(GNUInstallDirs)
enable_testing()

#-----------------------------------------------------------------

set( RES_VERSION_MAJOR 2 )   # Remember to update release notes whenever
set( RES_VERSION_MINOR 3 )   # you change the ERT_VERSION_MINOR or MAJOR
set( RES_VERSION_MICRO git ) # with "new in Ert Version X.X.X"!

# If the micro version is not integer, that should be interpreted as a
# development version leading towards version MAJOR.MINOR.0

execute_process(COMMAND date "+%Y-%m-%d %H:%M:%S" OUTPUT_VARIABLE RES_BUILD_TIME )
string(STRIP ${RES_BUILD_TIME} RES_BUILD_TIME)

find_package(Git)
if(GIT_FOUND)
   execute_process(
     COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
     OUTPUT_VARIABLE GIT_COMMIT
     OUTPUT_STRIP_TRAILING_WHITESPACE
   )

   execute_process(
     COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
     OUTPUT_VARIABLE GIT_COMMIT_SHORT
     OUTPUT_STRIP_TRAILING_WHITESPACE
   )
else()
    set( GIT_COMMIT "unknown (git not found!)")
    set( GIT_COMMIT_SHORT "unknown (git not found!)")
    message( WARNING "Git not found. Build will not contain git revision info." )
endif()

#-----------------------------------------------------------------

option( BUILD_TESTS         "Should the tests be built"             OFF)
option( BUILD_PYTHON        "Run py_compile on the python wrappers" OFF)
option( ENABLE_PYTHON       "Enable the python wrappers"            ON )
option( RST_DOC             "Build RST documentation"               OFF)
option( USE_RPATH           "Should we embed path to libraries"     ON )
option( INSTALL_ERT_LEGACY  "Install legacy ert code"               OFF)
option( ERT_LSF_SUBMIT_TEST "Build and run tests of LSF submit"     OFF)
set( SITE_CONFIG_FILE "${PROJECT_SOURCE_DIR}/share/site-config" CACHE FILEPATH "Path to global ERT Configuration file")

if (BUILD_PYTHON)
  if (NOT ENABLE_PYTHON)
    message(SEND_ERROR "The 'BUILD_PYTHON' option is deprecated - use 'ENABLE_PYTHON' - the two are inconsistent - exiting.")
  endif()
  message(WARNING "The 'BUILD_PYTHON' option is deprecated - use 'ENABLE_PYTHON=ON'")
endif()

if (USE_RPATH)
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif ()

#-----------------------------------------------------------------

find_package( ecl REQUIRED )

set(ERT_LSF_LIB_PATH     "" CACHE FILEPATH "Path to search for the LSF libraries")
set(ERT_LSF_INCLUDE_PATH "" CACHE FILEPATH "Path to search for the LSF header files")
find_path(LSF_HEADER_PATH lsf/lsf.h PATHS ${ERT_LSF_INCLUDE_PATH})
find_library(LSF_LIBRARY NAMES lsf PATHS ${ERT_LSF_LIB_PATH})

#-----------------------------------------------------------------

set(STATOIL_TESTDATA_ROOT "" CACHE PATH  "Root to Statoil internal testdata")
if (EXISTS ${STATOIL_TESTDATA_ROOT})
    set( LINK "${CMAKE_CURRENT_SOURCE_DIR}/test-data/Statoil" )
    if (EXISTS ${LINK})
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove "${LINK}")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${STATOIL_TESTDATA_ROOT}" "${LINK}")
    message(STATUS "Linking testdata: ${LINK} -> ${STATOIL_TESTDATA_ROOT}")
endif()

file(COPY bin/job_dispatch.py DESTINATION ${PROJECT_BINARY_DIR}/bin )
set( ERT_ROOT "${PROJECT_BINARY_DIR}" )
#-----------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif()

if (MSVC)
    add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4996" )
else ()
    set(CMAKE_C_FLAGS "-std=gnu99 -Wall -Wno-unknown-pragmas ${CMAKE_C_FLAGS}")
endif()

if (NOT BUILD_SHARED_LIBS)
    message(WARNING "Building python - forcing shared libs.")
    set(BUILD_SHARED_LIBS ON)
endif ()

set(INSTALL_GROUP ""
        CACHE STRING "Group to install as - blank to install as current group")
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)
#-----------------------------------------------------------------

find_package(Threads)
if (CMAKE_USE_PTHREADS_INIT)
    set(HAVE_PTHREAD TRUE)
endif ()

# feature tests
include(CheckFunctionExists)
check_function_exists( regexec ERT_HAVE_REGEXP )
#-----------------------------------------------------------------

add_subdirectory(libres_util)
add_subdirectory(libconfig)
add_subdirectory(libsched)
add_subdirectory(librms)
add_subdirectory(libanalysis)
add_subdirectory(libjob_queue)
add_subdirectory(libenkf)
add_subdirectory(python)
add_subdirectory(applications)

if (RST_DOC)
    add_subdirectory( docs )
endif()

install(DIRECTORY share/ DESTINATION share)
install(PROGRAMS bin/job_dispatch.py DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )

install(EXPORT res-config
        DESTINATION share/cmake/res
        NAMESPACE res::)
export(TARGETS res_util analysis config enkf job_queue rms sched
        NAMESPACE res::
        FILE resConfig.cmake)
export(PACKAGE res)
