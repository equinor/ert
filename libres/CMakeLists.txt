cmake_minimum_required(VERSION 3.12)
project(res C CXX)

option(BUILD_TESTS "Should the tests be built" OFF)
option(COVERAGE "Should binaries record coverage information" OFF)

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
# Set default CMAKE_BUILD_TYPE
# -----------------------------------------------------------------

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -Wall")
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

if(NOT SKBUILD)
  set(conan_opts REQUIRES pybind11/2.8.1)
endif()

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
  catch2/2.13.7
  cjson/1.7.15
  eigen/3.4.0
  fmt/8.0.1
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
find_package(cJSON REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)

find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  set(HAVE_PTHREAD TRUE)
endif()

# feature tests
include(CheckFunctionExists)
check_function_exists(regexec ERT_HAVE_REGEXP)

# -----------------------------------------------------------------
# Hack to get libres to compile without providing libecl on macOS
# -----------------------------------------------------------------

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS
      "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS
      "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
  set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS
      "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
  set(CMAKE_SHARED_MODULE_CREATE_CXX_FLAGS
      "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif()

# -----------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------

enable_testing()

add_subdirectory(lib)
add_subdirectory(old_tests)
add_subdirectory(tests)
