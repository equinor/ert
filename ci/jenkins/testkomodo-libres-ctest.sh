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

build_libres () {
    INSTALL=$WORKSPACE/install
    LIBRES_BUILD=$CI_SOURCE_ROOT/src/libres/build
    mkdir -p $LIBRES_BUILD
    pushd $LIBRES_BUILD
    KOMODO_PATH=/prog/res/komodo/${CI_KOMODO_RELEASE}
    cmake .. \
          -DCMAKE_PREFIX_PATH=$INSTALL \
          -DCMAKE_INSTALL_PREFIX=$INSTALL \
          -DBUILD_TESTS=ON \
          -DEQUINOR_TESTDATA_ROOT=/project/res-testdata/ErtTestData
    ninja
    popd
}

run_libres_ctest() {
    pushd $LIBRES_BUILD
    export ERT_SITE_CONFIG=${CI_SOURCE_ROOT}/src/ert_shared/share/ert/site-config

    ctest -j 6 -E Lint --output-on-failure
    popd
}

copy_test_files () {
     # libres
    mkdir -p ${CI_TEST_ROOT}/src/libres/res/fm/rms
    ln -s ${CI_SOURCE_ROOT}/src/res/fm/rms/rms_config.yml ${CI_TEST_ROOT}/src/libres/res/fm/rms/rms_config.yml
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
    build_libres
}

start_tests () {
    # build and run libres ctests
    run_libres_ctest
}
