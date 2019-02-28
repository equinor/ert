set(TEST_SOURCES
    __init__.py
    test_custom_kw.py
    test_custom_kw_config.py
    test_field_config.py
    test_gen_data.py
    test_gen_data_config.py
    test_gen_kw.py
    test_gen_kw_config.py
    test_gen_kw_config_equinor.py
    test_ext_param.py
    test_enkf_node.py
    test_enkf_config_node.py
    test_summary.py
)

add_python_package("python.tests.res.enkf.data" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/data "${TEST_SOURCES}" False)

python_config_test(tests.res.enkf.data.test_custom_kw.CustomKWTest LABELS SLOW_1)
python_config_test(tests.res.enkf.data.test_gen_kw.GenKwTest)
python_config_test(tests.res.enkf.data.test_ext_param.ExtParamTest)
python_config_test(tests.res.enkf.data.test_enkf_node.EnkfNodeTest)
python_config_test(tests.res.enkf.data.test_enkf_config_node.EnkfConfigNodeTest)
python_config_test(tests.res.enkf.data.test_gen_kw_config.GenKwConfigTest)
python_config_test(tests.res.enkf.data.test_summary.SummaryTest)
python_config_test(tests.res.enkf.data.test_gen_kw_config_equinor.GenKwConfigTest)
python_config_test(tests.res.enkf.data.test_gen_data.GenDataTest)
python_config_test(tests.res.enkf.data.test_gen_data_config.GenDataConfigTest)
