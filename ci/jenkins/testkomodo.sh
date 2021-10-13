
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

source_build_tools() {
    source /opt/rh/devtoolset-8/enable
    export PATH=/opt/rh/devtoolset-8/root/bin:$PATH
    pip install cmake
    python --version
    gcc --version
    cmake --version
}

build_libres () {
    INSTALL=$WORKSPACE/install
    LIBRES_BUILD=$CI_SOURCE_ROOT/libres/build
    mkdir -p $LIBRES_BUILD
    pushd $LIBRES_BUILD
    KOMODO_PATH=/prog/res/komodo/${CI_KOMODO_RELEASE}
    if [ -z "$CI_PR_RUN" ]
    then
        # In order to use the .so files from komodo, LD_LIBRARY_PATH must be set
        export LIBRES_LIB=$(find ${KOMODO_PATH}/root/ -name libres.so -exec dirname {} \;)
        export LIBECL_LIB=$(find ${KOMODO_PATH}/root/ -name libecl.so -exec dirname {} \;)
        export LD_LIBRARY_PATH=${LIBECL_LIB}:${LIBRES_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    fi
    cmake .. \
          -DCMAKE_PREFIX_PATH=$INSTALL \
          -DCMAKE_INSTALL_PREFIX=$INSTALL \
          -DBUILD_TESTS=ON \
          -DEQUINOR_TESTDATA_ROOT=/project/res-testdata/ErtTestData
    ninja
    popd
    # Remove built .so files when it is not a PR build
    if [ -z "$CI_PR_RUN" ]
    then
        find $LIBRES_BUILD -name *.so -delete
        find $INSTALL -name *.so -delete
    fi
}

run_libres_ctest() {
    pushd $LIBRES_BUILD
    export ERT_SITE_CONFIG=${CI_SOURCE_ROOT}/share/ert/site-config
    ctest -j 6 -E Lint --output-on-failure
    popd
}

copy_test_files () {
    cp -r ${CI_SOURCE_ROOT}/tests ${CI_TEST_ROOT}

    #ert
    ln -s ${CI_SOURCE_ROOT}/test-data ${CI_TEST_ROOT}/test-data
    ln -s ${CI_SOURCE_ROOT}/ert3_examples ${CI_TEST_ROOT}/ert3_examples

    # Trick ERT to find a fake source root
    mkdir ${CI_TEST_ROOT}/.git

    # libres
    mkdir -p ${CI_TEST_ROOT}/libres/res/fm/rms
    ln -s ${CI_SOURCE_ROOT}/res/fm/rms/rms_config.yml ${CI_TEST_ROOT}/libres/res/fm/rms/rms_config.yml
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/libres/lib
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/libres/bin

    ln -s ${CI_SOURCE_ROOT}/share ${CI_TEST_ROOT}/share
}

install_test_dependencies () {
    pip install -r dev-requirements.txt
}

install_package () {
    pip install . --no-deps
}

start_tests () {
    if [[ ${CI_KOMODO_RELEASE} =~ py27$  ]]
    then
        export PYTEST_QT_API=pyqt4v2
    fi
    export NO_PROXY=localhost,127.0.0.1

    # The existence of a running xvfb process will produce
    # a lock filgit ree for the default server and kill the run
    # Allow xvfb to find a new server
    pushd ${CI_TEST_ROOT}/tests/ert_tests
    xvfb-run -s "-screen 0 640x480x24" --auto-servernum python -m \
    pytest -k "not test_gui_load and not test_formatting" \
    -m "not requires_window_manager"
    popd

    pushd ${CI_TEST_ROOT}/tests/libres_tests
    ln -s /project/res-testdata/ErtTestData ${CI_TEST_ROOT}/test-data/Equinor
    export ECL_SKIP_SIGNAL=ON
    pytest                                                   \
        --ignore="tests/libres_tests/res/enkf/test_analysis_config.py"    \
        --ignore="tests/libres_tests/res/enkf/test_res_config.py"         \
        --ignore="tests/libres_tests/res/enkf/test_site_config.py"        \
        --ignore="tests/libres_tests/res/enkf/test_workflow_list.py"      \
        --ignore="tests/libres_tests/res/enkf/test_hook_manager.py"

    popd

    # build and run libres ctests
    ci_install_cmake
    ci_install_conan
    install_libecl
    build_libres
    run_libres_ctest

}
