set(TEST_SOURCES
    __init__.py
    test_ecl_config.py
    test_ecl_run.py
    test_rms_config.py
    test_rms_run.py
    test_shell.py
    test_templating.py
    test_fm_config.py
)

add_python_package("python.tests.res.fm"  ${PYTHON_INSTALL_PREFIX}/tests/res/fm "${TEST_SOURCES}" False)
