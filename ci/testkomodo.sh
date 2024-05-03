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
    pip install ".[dev]"
}

run_ert_with_opm () {
    pushd "${CI_TEST_ROOT}"

    mkdir ert_with_opm
    pushd ert_with_opm || exit 1

    cp "${CI_SOURCE_ROOT}/test-data/eclipse/SPE1.DATA" .

    cat > spe1_opm.ert << EOF
ECLBASE SPE1
DATA_FILE SPE1.DATA
RUNPATH realization-<IENS>/iter-<ITER>
NUM_REALIZATIONS 1
FORWARD_MODEL FLOW
EOF

    ert test_run spe1_opm.ert ||
        (
            # In case ert fails, print log files if they are there:
            cat realization-0/iter-0/STATUS  || true
            cat realization-0/iter-0/ERROR || true
            cat realization-0/iter-0/FLOW.stderr.0 || true
            cat realization-0/iter-0/FLOW.stdout.0 || true
            cat logs/ert-log* || true
        )
    popd
}

start_tests () {
    export NO_PROXY=localhost,127.0.0.1

    export ECL_SKIP_SIGNAL=ON

    pushd ${CI_TEST_ROOT}/tests

    python -m pytest -n auto --mpl --benchmark-disable --eclipse-simulator \
        --durations=0 -sv --dist loadgroup -m "not limit_memory" --max-worker-restart 0

    # Restricting the number of threads utilized by numpy to control memory consumption, as some tests evaluate memory usage and additional threads increase it.
    export OMP_NUM_THREADS=1

    python -m pytest -n 2 --durations=0 -m "limit_memory" --memray

    unset OMP_NUM_THREADS

    basetemp=$(mktemp -d -p $_ERT_TESTS_SHARED_TMP)
    pytest --timeout=3600 -v --$_ERT_TESTS_QUEUE_SYSTEM --basetemp="$basetemp" integration_tests/scheduler
    rm -rf "$basetemp" || true

    popd

    run_ert_with_opm
}
