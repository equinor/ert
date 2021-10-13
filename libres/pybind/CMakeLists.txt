# Scikit-Build does not add your site-packages to the search path automatically,
# so we need to add it _or_ the pybind11 specific directory here.
execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_clib MODULE init.cpp)
target_link_libraries(_clib PRIVATE res)
target_include_directories(_clib
PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/private-include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/private-include/ext/json>
        "${ECL_INCLUDE_DIRS}"
  )

install(TARGETS _clib DESTINATION res)
