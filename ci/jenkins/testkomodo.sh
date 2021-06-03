
copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    cp -r $CI_SOURCE_ROOT/test-data $CI_TEST_ROOT/test-data
    cp -r $CI_SOURCE_ROOT/ert3_examples $CI_TEST_ROOT/ert3_examples

    # Trick ERT to find a fake source root
    mkdir $CI_TEST_ROOT/.git

    # libres
    mkdir ${CI_TEST_ROOT}/libres
    mkdir -p ${CI_TEST_ROOT}/libres/res/fm/rms/
    ln -s ${CI_SOURCE_ROOT}/res/fm/rms/rms_config.yml ${CI_TEST_ROOT}/libres/res/fm/rms/rms_config.yml

    cp -r {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/libres/tests
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/libres/test-data
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/libres/lib
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/libres/bin

    ln -s $CI_SOURCE_ROOT/share ${CI_TEST_ROOT}/share
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
    xvfb-run -s "-screen 0 640x480x24" --auto-servernum python -m \
    pytest -k "not test_gui_load and not test_formatting" \
    -m "not requires_window_manager" --ignore ${CI_TEST_ROOT}/libres

    pushd ${CI_TEST_ROOT}/libres
    export ECL_SKIP_SIGNAL=ON
    pytest                                                   \
        --ignore="tests/res/enkf/test_analysis_config.py"    \
        --ignore="tests/res/enkf/test_res_config.py"         \
        --ignore="tests/res/enkf/test_site_config.py"        \
        --ignore="tests/res/enkf/test_workflow_list.py"      \
        --ignore="tests/res/enkf/test_hook_manager.py"       \
        --ignore="tests/legacy"                              \
        --ignore="tests/test_formatting.py"
    popd

}
