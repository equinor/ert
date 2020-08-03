
copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    cp -r $CI_SOURCE_ROOT/test-data $CI_TEST_ROOT/test-data
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
    # The existence of a running xvfb process will produce
    # a lock file for the default server and kill the run
    # Allow xvfb to find a new server
    xvfb-run --auto-servernum python -m pytest --ignore="tests/test_formatting.py" --ignore="tests/gui/test_gui_load.py"
}
