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
    export NO_PROXY=localhost,127.0.0.1

    pytest -m "not requires_window_manager" tests/
    popd
}
