# It is only necessary to change the start_tests function
SCRIPT_DIR=$(dirname "$0")
source $SCRIPT_DIR/testkomodo.sh

start_tests () {
    export NO_PROXY=localhost,127.0.0.1

    pushd ${CI_TEST_ROOT}/tests/ert_tests
    pytest --count=100 -x  tests/ert_tests/dark_storage
    pytest --count=100 -x  tests/ert_tests/ensemble_evaluator
    pytest --count=100 -x  tests/ert_tests/ert3
    popd
}
