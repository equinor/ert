
copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    cp -r $CI_SOURCE_ROOT/test-data $CI_TEST_ROOT/test-data
    cp -r $CI_SOURCE_ROOT/examples $CI_TEST_ROOT/examples
    # Trick ERT to find a fake source root
    mkdir $CI_TEST_ROOT/.git
}

install_test_dependencies () {
    pip install -r dev-requirements.txt
}

install_package () {
    pip install . --no-deps
}

start_tests () {
    if [[ ${CI_KOMODO_RELEASE} =~ py27$  ]]
    then
        export PYTEST_QT_API=pyqt4v2
    fi
    export NO_PROXY=localhost,127.0.0.1

    # The existence of a running xvfb process will produce
    # a lock filgit ree for the default server and kill the run
    # Allow xvfb to find a new server
    # requires_ert_storage - tests should be turned back when the storage solution is
    # integrated in Komodo.
    xvfb-run -s "-screen 0 640x480x24" --auto-servernum python -m \
    pytest -k "not test_gui_load and not test_formatting" \
    -m "not requires_window_manager and not requires_ert_storage"
}
