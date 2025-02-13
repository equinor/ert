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
    pip install ".[dev]"
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
    python -m pytest -n auto tests/everest -s
    return $?
}

# Clean up everest egg tmp folders
remove_one_week_old_temp_folders () {
    case "$1" in
        "azure")
            runner_root="/lustre1/users/f_scout_ci/egg_tests"
            ;;
        *)
            runner_root="/scratch/oompf/egg_tests"
            ;;
    esac
    old_directories=$(find $runner_root -maxdepth 1 -mtime +7 -user f_scout_ci -type d 2>/dev/null || true)
    if [[ -n "$old_directories" ]] ; then
        echo "Host: $(hostname -s), removing the following dirs: $old_directories"
        rm -rf "$old_directories" || true
    fi
}


make_run_path () {
    case "$1" in
        "azure")
            mktemp -d -p /lustre1/users/f_scout_ci/egg_tests
            ;;
        *)
            mktemp -d -p /scratch/oompf/egg_tests
            ;;
    esac
}

# Run everest egg test on LSF
run_everest_egg_test() {

    # Need to copy the egg test to a directory that is accessible by the LSF cluster
    LSF_RUNNER_ROOT=$(make_run_path "$CI_RUNNER_LABEL")
    mkdir -p "$LSF_RUNNER_ROOT"
    LSF_TMP_DIR=$(mktemp -d -p "$LSF_RUNNER_ROOT")
    chmod a+rx "$LSF_TMP_DIR"
    cp -r "${CI_SOURCE_ROOT}/test-data/everest/egg" "$LSF_TMP_DIR"
    pushd "${LSF_TMP_DIR}/egg"
    echo "LSF_TMP_DIR: $LSF_TMP_DIR"

    # Need to activate a komodo version available on LSF. Not a komodoenv
    disable_komodo
    # shellcheck source=/dev/null
    source "${_KOMODO_ROOT}/${_FULL_RELEASE_NAME}/enable"


    CONFIG="everest/model/config.yml"
    sed -i "s/name: local/name: lsf/g" "$CONFIG"
    cat "$CONFIG"
    export PATH=$PATH:/global/bin  # LSF executables

    everest run "$CONFIG"
    STATUS=$?
    popd

    remove_one_week_old_temp_folders "$CI_RUNNER_LABEL"

    return $STATUS
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

    run_everest_egg_test
    return_code_everest_egg_test=$?

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
    if [ "$return_code_everest_egg_test" -ne 0 ]; then
        echo "Everest egg tests failed."
        return_code_combined_tests=1
    fi
    return $return_code_combined_tests

}
