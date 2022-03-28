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
    pip install pytest-repeat
}

start_tests () {
    export NO_PROXY=localhost,127.0.0.1
    export ERT_SHOW_BACKTRACE=1

    ln -s /project/res-testdata/ErtTestData ${CI_TEST_ROOT}/test-data/Equinor

    pushd ${CI_TEST_ROOT}/tests/libres_tests
    # We want all of the pytest lines to execute, even if one of the fails.
    for i in `seq 1 1000`
    do
      pytest -svv res/enkf/data/test_gen_data_config.py || break;
    done
    popd
}
