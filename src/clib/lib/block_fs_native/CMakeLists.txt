pybind11_add_module(_block_fs_native MODULE module.cpp)

# -----------------------------------------------------------------
# Target: Python C Extension 'ert.storage.migration._block_fs_native'
# -----------------------------------------------------------------

target_link_libraries(
  _block_fs_native PRIVATE std::filesystem fmt::fmt cppitertools::cppitertools
                           ZLIB::ZLIB)
install(TARGETS _block_fs_native LIBRARY DESTINATION src/ert/storage/migration)
