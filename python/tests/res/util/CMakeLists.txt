set(TEST_SOURCES
    __init__.py
    test_ui_return.py
    test_log.py
    test_arg_pack.py
    test_res_log.py
    test_substitution_list.py
    test_version.py
    test_matrix.py
    test_path_fmt.py
    test_stat.py
    test_cthread_pool.py
)

add_python_package("python.tests.res.util" ${PYTHON_INSTALL_PREFIX}/tests/res/util "${TEST_SOURCES}" False)
