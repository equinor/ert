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

start_tests () {
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
    if [[ ! -z "${CI_PR_RUN:-}" ]]
    then
        pip install .
    fi

    copy_test_files

    install_test_dependencies
    start_tests
}
