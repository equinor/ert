set(TEST_SOURCES
    __init__.py
    test_ui_return.py
    test_log.py
    test_res_log.py
    test_substitution_list.py
    test_version.py
    test_path_fmt.py
)

add_python_package("python.tests.res.util" ${PYTHON_INSTALL_PREFIX}/tests/res/util "${TEST_SOURCES}" False)

addPythonTest(tests.res.util.test_log.LogTest)
addPythonTest(tests.res.util.test_res_log.ResLogTest)
addPythonTest(tests.res.util.test_substitution_list.SubstitutionListTest)
addPythonTest(tests.res.util.test_version.VersionTest)
addPythonTest(tests.res.util.test_path_fmt.PathFmtTest)
