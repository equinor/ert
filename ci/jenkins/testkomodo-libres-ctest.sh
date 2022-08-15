install_libecl () {
    INSTALL=$WORKSPACE/install
    mkdir -p $INSTALL
    LIBECL_ROOT=$WORKSPACE/libecl
    LIBECL_BUILD=$LIBECL_ROOT/build
    git clone https://github.com/equinor/libecl $LIBECL_ROOT
    mkdir -p $LIBECL_BUILD
    pushd $LIBECL_BUILD
    cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL -DCMAKE_BUILD_TYPE=RelWithDebInfo
    ninja install
    popd
}

build_ert_clib () {
    INSTALL=$WORKSPACE/install
    ERT_CLIB_BUILD=$CI_SOURCE_ROOT/src/clib/build
    mkdir -p $ERT_CLIB_BUILD
    pushd $ERT_CLIB_BUILD
    KOMODO_PATH=/prog/res/komodo/${CI_KOMODO_RELEASE}
    cmake .. \
          -DCMAKE_PREFIX_PATH=$INSTALL \
          -DCMAKE_INSTALL_PREFIX=$INSTALL \
          -DBUILD_TESTS=ON \
          -DEQUINOR_TESTDATA_ROOT=/project/res-testdata/ErtTestData
    ninja
    popd
}

run_ert_clib_tests() {
    pushd $ERT_CLIB_BUILD
    export ERT_SITE_CONFIG=${CI_SOURCE_ROOT}/src/ert/shared/share/ert/site-config

    ctest -j 6 -E Lint --output-on-failure
    popd
}

copy_test_files () {
    mkdir -p ${CI_TEST_ROOT}/src/clib/res/fm/rms
    ln -s ${CI_SOURCE_ROOT}/src/clib/_c_wrappers/fm/rms/rms_config.yml ${CI_TEST_ROOT}/src/clib/res/fm/rms/rms_config.yml
}

install_test_dependencies () {
    # empty to aviod running default install
    echo "no test deps"
}

install_package () {
    ci_install_cmake
    ci_install_conan

    python -m pip install pybind11
    install_libecl
    build_ert_clib
}

start_tests () {
    # build and run libres ctests
    run_ert_clib_tests
}
