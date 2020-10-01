_build_libres () {
    echo "Building libres"
    cmake -S . -B cmake-build -DBUILD_TESTS=ON
    cmake --build cmake-build
}

copy_test_files () {
    mkdir $CI_TEST_ROOT/.git
    mkdir -p $CI_TEST_ROOT/python/res/fm/rms/
    
    local PWD=$(pwd)
    cp -r {$PWD,$CI_TEST_ROOT}/python/tests
    ln -s {$PWD,$CI_TEST_ROOT}/test-data
    ln -s {$PWD,$CI_TEST_ROOT}/lib
    ln -s {$PWD,$CI_TEST_ROOT}/share
    ln -s {$PWD,$CI_TEST_ROOT}/bin
    ln -s {$PWD,$CI_TEST_ROOT}/python/res/fm/rms/rms_config.yml
}

install_package () {
    local PWD=$(pwd)
    export ERT_SITE_CONFIG=$PWD/share/ert/site-config
    export PYTHONPATH=$CI_SOURCE_ROOT/cmake-build/lib/python3.6/site-packages:${PYTHONPATH:-}
    export LD_LIBRARY_PATH=$CI_SOURCE_ROOT/cmake-build/lib:$CI_SOURCE_ROOT/cmake-build/lib64:${LD_LIBRARY_PATH:-}
}

start_tests () {
    echo "Running ctest"
    cd $CI_SOURCE_ROOT/cmake-build
    ctest --output-on-failure

    echo "Running pytest"
    cd $CI_TEST_ROOT/python
    export ECL_SKIP_SIGNAL=ON
    pytest                                                   \
        --ignore="tests/res/enkf/test_analysis_config.py"    \
        --ignore="tests/res/enkf/test_res_config.py"         \
        --ignore="tests/res/enkf/test_site_config.py"        \
        --ignore="tests/res/enkf/test_workflow_list.py"      \
        --ignore="tests/res/enkf/test_hook_manager.py"       \
        --ignore="tests/legacy"                              \
        --ignore="tests/test_formatting.py"
}

run_tests () {
    ci_install_cmake
    _build_libres
    if [ ! -z "${CI_PR_RUN:-}" ]
    then
        install_package
    else
        #removing built libs in order to ensure we are using libs from komodo
        rm -r cmake-build/lib64
    fi

    copy_test_files

    install_test_dependencies
    start_tests
}
