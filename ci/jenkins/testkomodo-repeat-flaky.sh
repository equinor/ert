copy_test_files () {
    cp -r ${CI_SOURCE_ROOT}/tests ${CI_TEST_ROOT}
    ln -s ${CI_SOURCE_ROOT}/test-data ${CI_TEST_ROOT}/test-data

    ln -s ${CI_SOURCE_ROOT}/src ${CI_TEST_ROOT}/src

    # Trick ERT to find a fake source root
    mkdir ${CI_TEST_ROOT}/.git

    # Keep pytest configuration:
    ln -s ${CI_SOURCE_ROOT}/pyproject.toml ${CI_TEST_ROOT}/pyproject.toml
}

install_test_dependencies () {
    pip install -r dev-requirements.txt
    pip install pytest-repeat
}

run_flaky_tests(){
    # If a developer would like to test a specific function, remove these
    # lines and include the specific tests one would like to run.
    # The internal job will use the testkomodo-repeat-flaky.sh from the branch
    # when running a PR job. Any periodically running jobs will use the version
    # on main - hence there will be no consequences for said jobs.
    # A test on the internal CI is activated by writing a comment with "test flaky please"
    # Requires that the user is allowed to run the tests.
    xvfb-run -s "-screen 0 640x480x24" --auto-servernum python -m \
    pytest tests/ -m "not requires_window_manager"
}

start_tests () {
    export NO_PROXY=localhost,127.0.0.1
    export ERT_SHOW_BACKTRACE=1

    pushd ${CI_TEST_ROOT}
    set +e

    let failures=0

    if [[ -z "${N_RUNS}" ]]; then
        n_runs=10
    else
        n_runs=${N_RUNS}
    fi

    for ((i = 0; i <= $n_runs; i++))
    do
        if ! run_flaky_tests; then
            ((failures +=1))
        fi
    done
    export N_FAILED_RUNS=$failures
    set -e
    popd
}
