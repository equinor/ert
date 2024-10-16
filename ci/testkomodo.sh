copy_test_files () {
    cp -r "${CI_SOURCE_ROOT}"/tests "${CI_TEST_ROOT}"
    cp -r "${CI_SOURCE_ROOT}"/docs "${CI_TEST_ROOT}"/docs
    ln -s "${CI_SOURCE_ROOT}"/test-data "${CI_TEST_ROOT}"/test-data

    ln -s "${CI_SOURCE_ROOT}"/src "${CI_TEST_ROOT}"/src

    # Trick ERT to find a fake source root
    mkdir "${CI_TEST_ROOT}"/.git

    # Keep pytest configuration:
    ln -s "${CI_SOURCE_ROOT}"/pyproject.toml "${CI_TEST_ROOT}"/pyproject.toml
}

install_test_dependencies () {
    pip install ".[dev, everest]"
}

run_ert_with_opm () {
    pushd "${CI_TEST_ROOT}"

    cp -r "${CI_SOURCE_ROOT}/test-data/ert/flow_example" ert_with_opm
    pushd ert_with_opm || exit 1

    ert test_run flow.ert ||
        (
            # In case ert fails, print log files if they are there:
            cat spe1_out/realization-0/iter-0/STATUS  || true
            cat spe1_out/realization-0/iter-0/ERROR || true
            cat spe1_out/realization-0/iter-0/FLOW.stderr.0 || true
            cat spe1_out/realization-0/iter-0/FLOW.stdout.0 || true
            cat logs/ert-log* || true
        )
    popd
}

run_everest_tests () {
    python -m pytest tests/everest -s \
    --ignore-glob "*test_visualization_entry*" \
     -m "not simulation_test and not ui_test"
    xvfb-run -s "-screen 0 640x480x24" --auto-servernum python -m pytest tests/everest -s -m "ui_test"
}

start_tests () {
    export NO_PROXY=localhost,127.0.0.1

    export ECL_SKIP_SIGNAL=ON

    pushd "${CI_TEST_ROOT}"/tests/ert

    set +e

    pytest --eclipse-simulator -n logical --show-capture=stderr -v --max-worker-restart 0 \
        -m "not limit_memory and not requires_window_manager" --benchmark-disable --dist loadgroup
    return_code_0=$?
    pytest --eclipse-simulator -v --mpl \
        -m "not limit_memory and requires_window_manager" --benchmark-disable
    return_code_1=$?

    # Restricting the number of threads utilized by numpy to control memory consumption, as some tests evaluate memory usage and additional threads increase it.
    export OMP_NUM_THREADS=1

    pytest -n 2 --durations=0 -m "limit_memory" --memray
    return_code_2=$?

    unset OMP_NUM_THREADS

    basetemp=$(mktemp -d -p "$_ERT_TESTS_SHARED_TMP")
    pytest --timeout=3600 -v --"$_ERT_TESTS_QUEUE_SYSTEM" --basetemp="$basetemp" unit_tests/scheduler
    return_code_3=$?
    rm -rf "$basetemp" || true

    popd

    run_ert_with_opm
    return_code_4=$?

    run_everest_tests
    return_code_5=$?
    set -e

    # We error if one or more returncodes are nonzero
    for code in $return_code_0 $return_code_1 $return_code_2 $return_code_3 $return_code_4 $return_code_5; do
        if [ "$code" -ne 0 ]; then
            echo "One or more tests failed."
            return 1
        fi
    done

}
