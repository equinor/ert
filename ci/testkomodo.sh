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
    pip install --group dev
}

run_ert_with_opm() {
    pushd "${CI_TEST_ROOT}" || exit 1

    cp -r "${CI_SOURCE_ROOT}/test-data/ert/flow_example" ert_with_opm
    pushd ert_with_opm || exit 1

    ert test_run --disable-monitoring flow.ert ||
        (
            # In case ert fails, print log files if they are there:
            cat spe1_out/realization-0/iter-0/STATUS || true
            cat spe1_out/realization-0/iter-0/ERROR || true
            cat spe1_out/realization-0/iter-0/FLOW.stderr.0 || true
            cat spe1_out/realization-0/iter-0/FLOW.stdout.0 || true
            cat logs/ert-log* || true
            exit 1
        )
    STATUS=$?
    popd || exit 1
    return "$STATUS"
}

# Run everest eightcells test on the cluster
run_everest_eightcells_test() {

    if [[ "$CI_RUNNER_LABEL" == "azure" ]]; then
        #RUNNER_ROOT="/lustre1/users/f_scout_ci/eightcells_tests"
        echo "Skip running everest eightcells test on azure for now"
        return 0
    elif [[ "$CI_RUNNER_LABEL" == "onprem" ]]; then
        RUNNER_ROOT="/scratch/oompf/eightcells_tests"
    else
        echo "Unsupported runner label: $CI_RUNNER_LABEL"
        return 1
    fi

    mkdir -p "$RUNNER_ROOT"

    EIGHTCELLS_RUNPATH=$(mktemp -d -p "$RUNNER_ROOT")

    # Need to copy the eightcells test to a directory that is accessible by all cluster members
    cp -r "${CI_SOURCE_ROOT}/test-data/everest/eightcells" "$EIGHTCELLS_RUNPATH"
    chmod -R a+rx "$EIGHTCELLS_RUNPATH"
    pushd "${EIGHTCELLS_RUNPATH}/eightcells" || exit 1
    echo "EIGHTCELLS_RUNPATH: $EIGHTCELLS_RUNPATH"

    disable_komodo
    # shellcheck source=/dev/null
    source "${_KOMODO_ROOT}/${_FULL_RELEASE_NAME}/enable"

    CONFIG="everest/model/config.yml"
    if [[ "$CI_RUNNER_LABEL" == "azure" ]]; then
        sed -i "s/name: local/name: torque\n    queue: permanent_8/g" "$CONFIG"
        export PATH=$PATH:/opt/pbs/bin
    elif [[ "$CI_RUNNER_LABEL" == "onprem" ]]; then
        sed -i "s/name: local/name: lsf/g" "$CONFIG"
        export PATH=$PATH:/global/bin
    fi

    everest run "$CONFIG" --skip-prompt --debug --disable-monitoring
    STATUS=$?
    popd || exit 1

    if [ $STATUS -ne 0 ]; then
        echo "Everest eightcells test failed. Running everest kill"
        everest kill "$CONFIG"
    fi

    # Clean up the temp folder removing folders older than 7 days
    find "$RUNNER_ROOT" -maxdepth 1 -mtime +7 -user f_scout_ci -type d -exec rm -r {} \;

    return $STATUS
}

start_tests() {
    export NO_PROXY=localhost,127.0.0.1
    export ERT_PYTEST_ARGS=--eclipse-simulator

    pushd "${CI_TEST_ROOT}"/tests/ert || exit 1

    if [ "$CI_SUBSYSTEM_TEST" == "ert" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile ert-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-gui-tests" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile ert-gui-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-cli-tests" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile ert-cli-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-unit-tests" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile ert-unit-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "everest" ]; then
      just -f "${CI_SOURCE_ROOT}"/justfile everest-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "everest-eightcells" ]; then
      run_everest_eightcells_test
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-limit-memory" ]; then
      # Restricting the number of threads utilized by numpy to control memory consumption, as some tests evaluate memory usage and additional threads increase it.
      export OMP_NUM_THREADS=1

      # Run ert tests that evaluates memory consumption
      just -f "${CI_SOURCE_ROOT}"/justfile ert-memory-tests
      return $?
    elif [ "$CI_SUBSYSTEM_TEST" == "ert-queue-system" ]; then
      basetemp=$(mktemp -d -p "$_ERT_TESTS_SHARED_TMP")
      pytest --timeout=3600 -v --"$_ERT_TESTS_QUEUE_SYSTEM" --basetemp="$basetemp" unit_tests/scheduler
      return_code_ert_scheduler_tests=$?
      rm -rf "$basetemp" || true
      return $return_code_ert_scheduler_tests
    elif [ "$CI_SUBSYSTEM_TEST" == "opm-integration" ]; then
      run_ert_with_opm
      return $?
    fi

    if [ -n "$CI_SUBSYSTEM_TEST" ]; then
      echo "Error: No argument for subsystem was provided."
      echo "Possible subsystems are specified with ert[<subsystem>]."
    else
      echo "Error: Variable $CI_SUBSYSTEM_TEST did not match any testable subsystem"
    fi
    echo "Possible subsystems are: ert, everest, everest-eightcells, ert-limit-memory, ert-queue-system, opm-integration"
    return 1
}
