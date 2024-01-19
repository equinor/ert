cmake_minimum_required(VERSION 3.19)
project(ert C CXX)

option(BUILD_TESTS "Should the tests be built" ON)
option(COVERAGE "Should binaries record coverage information" OFF)

if(NOT BUILD_TESTS)
  set(TESTS_EXCLUDE_FROM_ALL "EXCLUDE_FROM_ALL")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
     "${CMAKE_CURRENT_BINARY_DIR}")

if(NOT SKBUILD)
  message(
    WARNING
      "This CMakeLists.txt file should not be used directly.\n"
      "Use 'pip install ${CMAKE_SOURCE_DIR}' to install this as a Python package.\n"
      "Refer to the README for more information.")
endif()

# -----------------------------------------------------------------
# Detect pybind11
# -----------------------------------------------------------------

# Default to "python3" in user's local environment if PYTHON_EXECUTABLE is not
# set. This is preferable to CMake's `set(Python_EXECUTABLE ... CACHE ...)`
# because we avoid messing with future `FindPython.cmake` invocations.
if(DEFINED Python_EXECUTABLE)
  set(_python_executable "${Python_EXECUTABLE}")
else()
  set(_python_executable "python3")
endif()

execute_process(
  COMMAND "${_python_executable}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL
  LAST)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

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

if(COVERAGE)
  add_compile_options("--coverage")
  add_compile_options("-fPIC")
  add_link_options("--coverage")
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
    "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
    "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  # set(conan_opts ...)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # set(conan_opts ...)
else()
  message(
    WARNING "Unknown platform. Conan packages may not be configures correctly.")
endif()

conan_cmake_run(
  # Packages
  REQUIRES
  catch2/2.13.8
  eigen/3.4.0
  fmt/8.0.1
  zlib/1.3
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
# Dependencies
# -----------------------------------------------------------------

find_package(Eigen3 REQUIRED)
find_package(Filesystem REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)
find_package(ZLIB REQUIRED)

# feature tests
include(CheckFunctionExists)
check_function_exists(regexec ERT_HAVE_REGEXP)

# -----------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------

add_subdirectory(lib)
add_subdirectory(lib/block_fs_native)

if(BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(tests)
