build() {
    mkdir build
    cmake -S src/clib -B build -DCMAKE_BUILD_TYPE=Debug \
          -DEQUINOR_TESTDATA_ROOT=/project/res-testdata/ErtTestData
    cmake --build build
}

copy_test_files () {
    mkdir -p ${CI_TEST_ROOT}/src/clib/res/fm/rms
}

install_test_dependencies () {
    # empty to avoid running default install
    :
}

install_package () {
    ci_install_cmake
    ci_install_conan

    python -m pip install pybind11
    build
}

start_tests () {
    export ERT_SITE_CONFIG=${CI_SOURCE_ROOT}/src/ert/shared/share/ert/site-config
    ctest -j $(nproc) -E Lint --output-on-failure
}
