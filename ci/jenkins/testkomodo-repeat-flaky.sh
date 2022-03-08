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

start_tests () {
    export NO_PROXY=localhost,127.0.0.1

    pushd ${CI_TEST_ROOT}/tests/ert_tests
    pytest --count=100 -x  tests/ert_tests/dark_storage
    pytest --count=100 -x  tests/ert_tests/ensemble_evaluator
    pytest --count=100 -x  tests/ert_tests/ert3
    popd

}
