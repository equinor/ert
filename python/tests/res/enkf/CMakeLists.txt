set(TEST_SOURCES
    __init__.py
    test_active_list.py
    test_analysis_config.py
    test_analysis_iter_config.py
    test_block_obs.py
    test_custom_kw_config_set.py
    test_ecl_config.py
    test_es_update.py
    test_enkf.py
    test_enkf_fs.py
    test_enkf_fs_manager1.py
    test_enkf_fs_manager2.py
    test_enkf_library.py
    test_enkf_load_results_manually.py
    test_enkf_obs.py
    test_ert_context.py
    test_ert_log.py
    test_gen_obs.py
    test_labscale.py
    test_local_obsdata_node.py
    test_local_config.py
    test_meas_block.py
    test_meas_data.py
    test_obs_block.py
    test_obs_data.py
    test_state_map.py
    test_summary_key_matcher.py
    test_summary_key_set.py
    test_summary_obs.py
    test_time_map.py
    test_update.py
    test_ensemble_config.py
    test_deprecation.py
    test_field_config.py
    test_field_export.py
    test_hook_workflow.py
    test_forward_load_context.py
    test_runpath_list.py
)

add_python_package("python.tests.res.enkf" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf "${TEST_SOURCES}" False)
addPythonTest(res.enkf.es_update tests.res.enkf.test_es_update.ESUpdateTest)
addPythonTest(res.enkf.forward_load_context tests.res.enkf.test_forward_load_context.ForwardLoadContextTest)
addPythonTest(res.enkf.deprecation tests.res.enkf.test_deprecation.DeprecationTest)
addPythonTest(res.enkf.meas_block tests.res.enkf.test_meas_block.MeasBlockTest)
addPythonTest(res.enkf.meas_data tests.res.enkf.test_meas_data.MeasDataTest)
addPythonTest(res.enkf.obs_data tests.res.enkf.test_obs_data.ObsDataTest)
addPythonTest(res.enkf.gen_obs tests.res.enkf.test_gen_obs.GenObsTest)
addPythonTest(res.enkf.summary_obs tests.res.enkf.test_summary_obs.SummaryObsTest)
addPythonTest(res.enkf.block_obs tests.res.enkf.test_block_obs.BlockObsTest)
addPythonTest(res.enkf.obs_block tests.res.enkf.test_obs_block.ObsBlockTest)
addPythonTest(res.enkf.ert_log tests.res.enkf.test_ert_log.ErtLogTest)
addPythonTest(res.enkf.analysis_config tests.res.enkf.test_analysis_config.AnalysisConfigTest)
addPythonTest(res.enkf.analysis_iter_config tests.res.enkf.test_analysis_iter_config.AnalysisIterConfigTest)
addPythonTest(res.enkf.state_map tests.res.enkf.test_state_map.StateMapTest)
addPythonTest(res.enkf.custom_kw_config_set tests.res.enkf.test_custom_kw_config_set.CustomKWConfigSetTest)
addPythonTest(res.enkf.summary_key_matcher tests.res.enkf.test_summary_key_matcher.SummaryKeyMatcherTest)
addPythonTest(res.enkf.time_map tests.res.enkf.test_time_map.TimeMapTest)
addPythonTest(res.enkf.active_list tests.res.enkf.test_active_list.ActiveListTest)
addPythonTest(res.enkf.local_obsdata_node tests.res.enkf.test_local_obsdata_node.LocalObsdataNodeTest)
addPythonTest(res.enkf.local_config tests.res.enkf.test_local_config.LocalConfigTest)
addPythonTest(res.enkf.ensemble_config tests.res.enkf.test_ensemble_config.EnsembleConfigTest)
addPythonTest(res.enkf.hook_workflow tests.res.enkf.test_hook_workflow.HookWorkFlowTest)
addPythonTest(res.enkf.runpath_list tests.res.enkf.test_runpath_list.RunpathListTest)
addPythonTest(res.enkf.field_config tests.res.enkf.test_field_config.FieldConfigTest)

if (STATOIL_TESTDATA_ROOT)
  addPythonTest(res.enkf.enkf tests.res.enkf.test_enkf.EnKFTest LABELS StatoilData)
  addPythonTest(res.enkf.enkf_obs tests.res.enkf.test_enkf_obs.EnKFObsTest LABELS StatoilData)
  addPythonTest(res.enkf.enkf_load_results_manually tests.res.enkf.test_enkf_load_results_manually.LoadResultsManuallyTest LABELS StatoilData)
  addPythonTest(res.enkf.enkf_fs tests.res.enkf.test_enkf_fs.EnKFFSTest LABELS StatoilData)
  addPythonTest(res.enkf.enkf_fs_manager1 tests.res.enkf.test_enkf_fs_manager1.EnKFFSManagerTest1 LABELS StatoilData)
  addPythonTest(res.enkf.enkf_fs_manager2 tests.res.enkf.test_enkf_fs_manager2.EnKFFSManagerTest2 LABELS StatoilData)
  addPythonTest(res.enkf.enkf_library tests.res.enkf.test_enkf_library.EnKFLibraryTest)
  addPythonTest(res.enkf.summary_key_set tests.res.enkf.test_summary_key_set.SummaryKeySetTest LABELS StatoilData)
  addPythonTest(res.enkf.ecl_config tests.res.enkf.test_ecl_config.EclConfigTest LABELS StatoilData)
  addPythonTest(res.enkf.ert_test_context tests.res.enkf.test_ert_context.ErtTestContextTest LABELS StatoilData)
  addPythonTest(res.enkf.field_export tests.res.enkf.test_field_export.FieldExportTest LABELS StatoilData)

  # The following two tests involve loading the rml module as part of
  # the bootup process. That fails with a dlopen( ) error I just can
  # not understand; when the libraries have been installed with 'make
  # install' the tests seem to work.

  # addPythonTest(ert.enkf.labscale tests.res.enkf.test_labscale.LabScaleTest LABELS StatoilData)
  # addPythonTest(ert.enkf.update tests.res.enkf.test_update.UpdateTest LABELS StatoilData)
endif()

add_subdirectory(data)
add_subdirectory(export)
add_subdirectory(plot)
