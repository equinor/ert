cmake_minimum_required(VERSION 3.6.1)
project(res C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" "${CMAKE_CURRENT_BINARY_DIR}")

if(NOT SKBUILD)
  message(WARNING "This CMakeLists.txt file should not be used directly.\n"
    "Use 'pip install ${CMAKE_SOURCE_DIR}' to install this as a Python package.\n"
    "Refer to the README for more information.")
endif()

# -----------------------------------------------------------------
# Set default CMAKE_BUILD_TYPE
# -----------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING "CMake build type" FORCE)
  message(
    STATUS
      "Setting CMake build type to ${CMAKE_BUILD_TYPE} as none was specified")

  # Set possible values for use in cmake-gui
  set(CACHE
      CMAKE_BUILD_TYPE
      PROPERTY
      STRINGS
      "Debug"
      "Release"
      "MinSizeRel"
      "RelWithDebInfo")
endif()

# -----------------------------------------------------------------
# Conan: C++ package manager
# https://docs.conan.io/en/latest/howtos/cmake_launch.html
# -----------------------------------------------------------------

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(
    STATUS
      "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(
    DOWNLOAD
    "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
    "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  # set(conan_opts ...)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # set(conan_opts ...)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # set(conan_opts ...)
else()
  message(
    WARNING "Unknown platform. Conan packages may not be configures correctly.")
endif()

conan_cmake_run(
  # Packages
  REQUIRES
  catch2/2.13.7
  # Options
  OPTIONS
  catch2:with_main=True
  ${conan_opts}
  # Force cppstd to be the same as this CMakeLists.txt's
  SETTINGS
  compiler.cppstd=${CMAKE_CXX_STANDARD}
  # Build from source if there are no pre-compiled binaries
  BUILD
  missing
  GENERATORS
  cmake_find_package)

# -----------------------------------------------------------------

if(NOT DEFINED CMAKE_MACOSX_RPATH)
  # There is some weirdness around this variable, the default value is different depending on
  # the cmake version, see policy CMP0042.
  # A more explicit way to treat this would be `cmake_policy(SET CMP0042 NEW)` but that would
  # fail on CMake 2.8.12 (the variable exists but the policy doesn't)
  set(CMAKE_MACOSX_RPATH ON)
endif()

include(GNUInstallDirs)
enable_testing()

#-----------------------------------------------------------------

set(RES_VERSION "0.0.0-unset" CACHE STRING "libres version")

string( REGEX REPLACE "^([^.]+)\\.([^.]+)\\.(.+)$" "\\1" RES_VERSION_MAJOR "${RES_VERSION}")
string( REGEX REPLACE "^([^.]+)\\.([^.]+)\\.(.+)$" "\\2" RES_VERSION_MINOR "${RES_VERSION}")
string( REGEX REPLACE "^([^.]+)\\.([^.]+)\\.(.+)$" "\\3" RES_VERSION_MICRO "${RES_VERSION}")

message( STATUS "libres version: ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR}.${RES_VERSION_MICRO}" )

#-----------------------------------------------------------------

option( BUILD_TESTS         "Should the tests be built"             OFF)
option( USE_RPATH           "Should we embed path to libraries"     ON )
option( ERT_LSF_SUBMIT_TEST "Build and run tests of LSF submit"     OFF)

if(NOT SKBUILD)
  # We let setuptools handle installing share/ert normally.

  set( SHARE_DIR "../ert_shared/share/ert")
  # If the SITE_CONFIG_FILE is not set as -DSITE_CONFIG_FILE switch when invoking
  # cmake we set it here to location of the to-be-installed site-config file. That
  # implies that when testing the binary will have embedded a path to non-existent
  # or stale version of the site configuration file. Irrespective of the value of
  # SITE_CONFIG_FILE the environment variable ERT_SITE_CONFIG will be set to point
  # to the in-source version of the site-config file for testing.
  #
  # The variable SITE_CONFIG_FILE can be set by the user, if that is set it will
  # take presedence. If the user has not explicitly set the SITE_CONFIG_FILE variable
  # we will use the default path based on the install prefix.
  set( SITE_CONFIG_FILE "${CMAKE_INSTALL_PREFIX}/${SHARE_DIR}/site-config" CACHE FILEPATH "Path to global ERT Configuration file")
  message(STATUS "The path ${SITE_CONFIG_FILE} will be compiled into the libres library as the location of the site configuration file")

  if (USE_RPATH)
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif ()
endif()

#-----------------------------------------------------------------

set(ERT_LSF_LIB_PATH     "" CACHE FILEPATH "Path to search for the LSF libraries")
set(ERT_LSF_INCLUDE_PATH "" CACHE FILEPATH "Path to search for the LSF header files")
find_path(LSF_HEADER_PATH lsf/lsf.h PATHS ${ERT_LSF_INCLUDE_PATH})
find_library(LSF_LIBRARY NAMES lsf PATHS ${ERT_LSF_LIB_PATH})

#-----------------------------------------------------------------

set(EQUINOR_TESTDATA_ROOT "" CACHE PATH  "Root to Equinor internal testdata")
if (EXISTS ${EQUINOR_TESTDATA_ROOT})
    set( LINK "${CMAKE_CURRENT_SOURCE_DIR}/../test-data/Equinor" )
    if (EXISTS ${LINK})
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove "${LINK}")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${EQUINOR_TESTDATA_ROOT}" "${LINK}")
    message(STATUS "Linking testdata: ${LINK} -> ${EQUINOR_TESTDATA_ROOT}")
endif()

file(COPY ../job_runner/job_dispatch.py
  DESTINATION ${PROJECT_BINARY_DIR}/bin
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)
set( ERT_ROOT "${PROJECT_BINARY_DIR}" )

#-----------------------------------------------------------------

find_package(Filesystem REQUIRED)

find_package(Threads)
if (CMAKE_USE_PTHREADS_INIT)
    set(HAVE_PTHREAD TRUE)
endif ()

# feature tests
include(CheckFunctionExists)
check_function_exists( regexec ERT_HAVE_REGEXP )

#-----------------------------------------------------------------

# This is to enable our dynamic lookup hack
if (APPLE)
    set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
    set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
    set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
    set(CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif ()

#-----------------------------------------------------------------
# install_example() is a small utility function which is used to install an
# example file. If a file with the same name already exists in the installation
# directory the example file will not be reinstalled on top.
#
# The decision whether to install the example file or not is evaluated at
# configure - time; i.e. there is a race open between configure time and
# installation. Should ideally use the INSTALL(CODE ...) functionality to defer
# the decision to installation time.

function(install_example src_file destination)

  set (_full_destination "${CMAKE_INSTALL_PREFIX}/${destination}")
  if (EXISTS "${_full_destination}/${src_file}")
    message(STATUS "File: ${_full_destination}/${src_file} already exists - will not be updated")
  else()
    message(STATUS "An example ${src_file} will be installed as ${_full_destination}")
    install(FILES ${src_file} DESTINATION "${_full_destination}")
  endif()

endfunction()


add_subdirectory(lib)
add_subdirectory(tests)
