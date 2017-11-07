set(TEST_SOURCES
    __init__.py
    test_custom_kw.py
    test_custom_kw_config.py
    test_field_config.py
    test_gen_data.py
    test_gen_data_config.py
    test_gen_kw.py
    test_gen_kw_config_statoil.py
    test_ext_param.py
    test_enkf_node.py
    test_enkf_config_node.py
)

add_python_package("python.tests.res.enkf.data" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf/data "${TEST_SOURCES}" False)

addPythonTest(tests.res.enkf.data.test_custom_kw.CustomKWTest LABELS SLOW_1)
addPythonTest(tests.res.enkf.data.test_gen_kw.GenKwTest)
addPythonTest(tests.res.enkf.data.test_ext_param.ExtParamTest)
addPythonTest(tests.res.enkf.data.test_enkf_node.EnkfNodeTest)
addPythonTest(tests.res.enkf.data.test_enkf_config_node.EnkfConfigNodeTest)

if (STATOIL_TESTDATA_ROOT)
  addPythonTest(tests.res.enkf.data.test_gen_kw_config_statoil.GenKwConfigTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.data.test_gen_data.GenDataTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.data.test_gen_data_config.GenDataConfigTest LABELS StatoilData)
endif()
