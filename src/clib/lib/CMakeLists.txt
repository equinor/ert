pybind11_add_module(
  _clib
  SHARED
  python/init.cpp
  python/logging.cpp
  enkf/enkf_obs.cpp
  enkf/read_summary.cpp
  enkf/row_scaling.cpp)

# -----------------------------------------------------------------
# Target: Python C Extension 'ert._clib'
# -----------------------------------------------------------------

target_link_libraries(_clib PUBLIC std::filesystem fmt::fmt Eigen3::Eigen
                                   Threads::Threads)
target_include_directories(_clib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                        "${ECL_INCLUDE_DIRS}")

set_target_properties(_clib PROPERTIES CXX_VISIBILITY_PRESET "default")
install(TARGETS _clib LIBRARY DESTINATION src/ert)

# -----------------------------------------------------------------
# Target: 'libert.so' for use in tests
# -----------------------------------------------------------------

if(BUILD_TESTS)
  add_library(ert SHARED $<TARGET_OBJECTS:_clib>)
  target_link_libraries(ert _clib pybind11::embed resdata)
  fix_install_names(ert)
endif()
