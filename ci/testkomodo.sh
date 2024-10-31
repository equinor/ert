copy_test_files() {
    cp -r "${CI_SOURCE_ROOT}"/tests "${CI_TEST_ROOT}"
    cp -r "${CI_SOURCE_ROOT}"/docs "${CI_TEST_ROOT}"/docs
    ln -s "${CI_SOURCE_ROOT}"/test-data "${CI_TEST_ROOT}"/test-data

    ln -s "${CI_SOURCE_ROOT}"/src "${CI_TEST_ROOT}"/src

    # Trick ERT to find a fake source root
    mkdir "${CI_TEST_ROOT}"/.git

    # Keep pytest configuration:
    ln -s "${CI_SOURCE_ROOT}"/pyproject.toml "${CI_TEST_ROOT}"/pyproject.toml
}

install_test_dependencies() {
    pip install ".[dev, everest]"
}

run_ert_with_opm() {
    pushd "${CI_TEST_ROOT}"

    cp -r "${CI_SOURCE_ROOT}/test-data/ert/flow_example" ert_with_opm
    pushd ert_with_opm || exit 1

    ert test_run flow.ert ||
        (
            # In case ert fails, print log files if they are there:
            cat spe1_out/realization-0/iter-0/STATUS || true
            cat spe1_out/realization-0/iter-0/ERROR || true
            cat spe1_out/realization-0/iter-0/FLOW.stderr.0 || true
            cat spe1_out/realization-0/iter-0/FLOW.stdout.0 || true
            cat logs/ert-log* || true
        )
    popd
}

run_everest_tests() {
    python -m pytest tests/everest -s \
    --ignore-glob "*test_visualization_entry*" \
     -m "not requires_eclipse"

}

start_tests() {
    export NO_PROXY=localhost,127.0.0.1

    export ECL_SKIP_SIGNAL=ON

    pushd "${CI_TEST_ROOT}"/tests/ert

    set +e

    # Run all ert tests except tests evaluating memory consumption and tests requiring windows manager (GUI tests)
    pytest --eclipse-simulator -n auto --show-capture=stderr -v --max-worker-restart 0 \
        -m "not limit_memory and not requires_window_manager" --benchmark-disable --dist loadgroup
    return_code_ert_main_tests=$?

    # Run all ert tests requiring windows manager (GUI tests) except tests evaluating memory consumption
    pytest --eclipse-simulator -v --mpl \
        -m "not limit_memory and requires_window_manager" --benchmark-disable
    return_code_ert_gui_tests=$?

    # Restricting the number of threads utilized by numpy to control memory consumption, as some tests evaluate memory usage and additional threads increase it.
    export OMP_NUM_THREADS=1

    # Run ert tests that evaluates memory consumption
    pytest -n 2 --durations=0 -m "limit_memory" --memray
    return_code_ert_memory_consumption_tests=$?

    unset OMP_NUM_THREADS

    # Run ert scheduler tests on the actual cluster (defined by $_ERT_TESTS_QUEUE_SYSTEM)
    basetemp=$(mktemp -d -p "$_ERT_TESTS_SHARED_TMP")
    pytest --timeout=3600 -v --"$_ERT_TESTS_QUEUE_SYSTEM" --basetemp="$basetemp" unit_tests/scheduler
    return_code_ert_scheduler_tests=$?
    rm -rf "$basetemp" || true

    popd

    run_ert_with_opm
    return_code_opm_integration_test=$?

    run_everest_tests
    return_code_everest_tests=$?
    set -e

    return_code_combined_tests=0
    # We error if one or more returncodes are nonzero
    if [ "$return_code_ert_main_tests" -ne 0 ]; then
        echo "One or more ERT tests failed."
        return_code_combined_tests=1
    fi
    if [ "$return_code_ert_gui_tests" -ne 0 ]; then
        echo "One or more ERT GUI tests failed."
        return_code_combined_tests=1
    fi
    if [ "$return_code_ert_memory_consumption_tests" -ne 0 ]; then
        echo "One or more ERT memory consumption tests failed."
        return_code_combined_tests=1
    fi
    if [ "$return_code_ert_scheduler_tests" -ne 0 ]; then
        echo "One or more ERT scheduler tests failed."
        return_code_combined_tests=1
    fi
    if [ "$return_code_opm_integration_test" -ne 0 ]; then
        echo "The ERT OPM integration test failed."
        return_code_combined_tests=1
    fi
    if [ "$return_code_everest_tests" -ne 0 ]; then
        echo "One or more Everest tests failed."
        return_code_combined_tests=1
    fi
    return $return_code_combined_tests

}
