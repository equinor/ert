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

    pushd ${CI_TEST_ROOT}/tests/ert_tests
    # We want all of the pytest lines to execute, even if one of the fails.
    set +e
    pytest --count=1000 -x shared/test_port_handler.py
    pytest --count=100 -x dark_storage/test_http_endpoints.py
    pytest --count=100 -x dark_storage/test_api_compatibility.py
    pytest --count=100 -x ensemble_evaluator/test_ensemble_legacy.py

    # engine, services and status are integration tests and time consuming,
    # estimated a several hours just for these
    pytest --count=30 -x ert3/engine
    pytest --count=30 -x services
    pytest --count=30 -x status
    pytest --count=30 -x ert3/algorithms/test_fast.py # test fast is not fast
    # The following should be added, but as a subclass of UnitTest TestCase it
    # is not possible to run with --count (they will only run once)
    # tests/libres_tests/res/enkf/plot/test_plot_data.py
    # tests/libres_tests/res/simulator/test_batch_sim.py
    # tests/libres_tests/res/enkf/data/test_gen_kw_config.py
    set -e
    popd
}
