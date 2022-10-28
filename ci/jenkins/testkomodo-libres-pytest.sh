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
}

install_package () {
    pip install . --no-deps
}

start_tests () {
    pushd ${CI_TEST_ROOT}/tests/
    ln -s /project/res-testdata/ErtTestData ${CI_TEST_ROOT}/test-data/Equinor
    export ECL_SKIP_SIGNAL=ON
    pytest -m "requires_eclipse" --eclipse-simulator

    popd
}
