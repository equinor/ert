copy_test_files () {
    cp -r ${CI_SOURCE_ROOT}/tests ${CI_TEST_ROOT}

    #ert
    ln -s ${CI_SOURCE_ROOT}/test-data ${CI_TEST_ROOT}/test-data

    # Trick ERT to find a fake source root
    mkdir ${CI_TEST_ROOT}/.git

    # libres
    mkdir -p ${CI_TEST_ROOT}/src/clib/res/fm/rms
    ln -s ${CI_SOURCE_ROOT}/src/ert/_c_wrappers/fm/rms/rms_config.yml ${CI_TEST_ROOT}/src/clib/res/fm/rms/rms_config.yml
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/src/clib/lib
    ln -s {$CI_SOURCE_ROOT,$CI_TEST_ROOT}/src/clib/bin

    ln -s ${CI_SOURCE_ROOT}/share ${CI_TEST_ROOT}/share

    # Keep pytest configuration:
    ln -s ${CI_SOURCE_ROOT}/pyproject.toml ${CI_TEST_ROOT}/pyproject.toml

}

install_test_dependencies () {
    pip install -r dev-requirements.txt
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

}
