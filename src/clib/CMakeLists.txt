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

# Force old pre-C++11 ABI on Linux
#
# libecl that is distributed on PyPI is build using manylinux2014 at time of
# writing. Manylinux2014 is based on CentOS 7, which is in turn based on RHEL 7.
# The latter hard-code the ABI to be pre-C++11 with no option to enable the new
# ABI. ERT links directly to libecl and therefore has to use the same ABI.
# Otherwise we get linker errors because `std::basic_string` and
# `std::__cxx11::basic_string` are distinct types.
#
# This line should be removed either when we stop linking to libecl directly, we
# use a newer manylinux (RHEL 8 and up supports the new ABI), or the heat death
# of the universe occurs.
#
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
add_compile_definitions("_GLIBCXX_USE_CXX11_ABI=0")

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
# Detect libecl
# -----------------------------------------------------------------

execute_process(
  COMMAND "${_python_executable}" -c "import ecl; print(ecl.get_include())"
  OUTPUT_VARIABLE ECL_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL
  LAST)

execute_process(
  COMMAND "${_python_executable}" -c
          "import ecl; print(ecl.EclPrototype.lib._name)"
  OUTPUT_VARIABLE ECL_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL
  LAST)

add_library(ecl SHARED IMPORTED GLOBAL)
set_target_properties(ecl PROPERTIES IMPORTED_LOCATION "${ECL_LIBRARY}"
                                     IMPORTED_NO_SONAME TRUE)

function(fix_install_names target)
  # CMake doesn't let us link to absolute paths on macOS. This means that
  # regardless of where libecl is, the OS will expect to find it in either
  # "/usr/lib" or "/usr/local/lib". This function will force a full path.

  if(APPLE)
    add_custom_command(
      TARGET "${target}"
      POST_BUILD
      COMMAND "${CMAKE_INSTALL_NAME_TOOL}" -change libecl.dylib
              "${ECL_LIBRARY}" $<TARGET_FILE:${target}>)

    # PyPI 'ecl' ships with versioned shared libraries, and thus Clang will
    # prefer libecl.2.dylib over libecl.dylib. This is a bug that should be
    # resolved, and the following is a workaround.
    add_custom_command(
      TARGET "${target}"
      POST_BUILD
      COMMAND "${CMAKE_INSTALL_NAME_TOOL}" -change libecl.2.dylib
              "${ECL_LIBRARY}" $<TARGET_FILE:${target}>)
  endif()
endfunction()

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
  cjson/1.7.16
  cppitertools/2.1
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
find_package(cJSON REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)
find_package(cppitertools REQUIRED)
find_package(ZLIB REQUIRED)

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

add_subdirectory(lib)
add_subdirectory(lib/block_fs_native)

if(BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(tests)
