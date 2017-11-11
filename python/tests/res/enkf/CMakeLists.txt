set(TEST_SOURCES
    __init__.py
    test_simulation_batch.py
    test_model_config.py 
    test_active_list.py
    test_analysis_config.py
    test_analysis_iter_config.py
    test_block_obs.py
    test_custom_kw_config_set.py
    test_ecl_config.py
    test_es_update.py
    test_enkf.py
    test_enkf_transfer_env.py
    test_enkf_fs.py
    test_enkf_fs_manager1.py
    test_enkf_fs_manager2.py
    test_enkf_library.py
    test_enkf_load_results_manually.py
    test_enkf_obs.py
    test_enkf_runpath.py
    test_ert_context.py
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
    test_runpath_list_ert.py
    test_queue_config.py
    test_site_config.py
    test_res_config.py
    test_log_config.py
    test_run_context.py
    test_data_kw_define.py
    test_programmatic_res_config.py
)

add_python_package("python.tests.res.enkf" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf "${TEST_SOURCES}" False)

addPythonTest(tests.res.enkf.test_data_kw_define.DataKWTest)
addPythonTest(tests.res.enkf.test_es_update.ESUpdateTest)
addPythonTest(tests.res.enkf.test_site_config.SiteConfigTest)
addPythonTest(tests.res.enkf.test_res_config.ResConfigTest)
addPythonTest(tests.res.enkf.test_forward_load_context.ForwardLoadContextTest)
addPythonTest(tests.res.enkf.test_deprecation.DeprecationTest)
addPythonTest(tests.res.enkf.test_meas_block.MeasBlockTest)
addPythonTest(tests.res.enkf.test_meas_data.MeasDataTest)
addPythonTest(tests.res.enkf.test_obs_data.ObsDataTest)
addPythonTest(tests.res.enkf.test_gen_obs.GenObsTest)
addPythonTest(tests.res.enkf.test_summary_obs.SummaryObsTest)
addPythonTest(tests.res.enkf.test_block_obs.BlockObsTest)
addPythonTest(tests.res.enkf.test_obs_block.ObsBlockTest)
addPythonTest(tests.res.enkf.test_analysis_config.AnalysisConfigTest)
addPythonTest(tests.res.enkf.test_analysis_iter_config.AnalysisIterConfigTest)
addPythonTest(tests.res.enkf.test_state_map.StateMapTest)
addPythonTest(tests.res.enkf.test_custom_kw_config_set.CustomKWConfigSetTest)
addPythonTest(tests.res.enkf.test_summary_key_matcher.SummaryKeyMatcherTest)
addPythonTest(tests.res.enkf.test_time_map.TimeMapTest)
addPythonTest(tests.res.enkf.test_active_list.ActiveListTest)
addPythonTest(tests.res.enkf.test_local_obsdata_node.LocalObsdataNodeTest)
addPythonTest(tests.res.enkf.test_local_config.LocalConfigTest)
addPythonTest(tests.res.enkf.test_ensemble_config.EnsembleConfigTest)
addPythonTest(tests.res.enkf.test_hook_workflow.HookWorkFlowTest)
addPythonTest(tests.res.enkf.test_runpath_list.RunpathListTest)
addPythonTest(tests.res.enkf.test_field_config.FieldConfigTest)
addPythonTest(tests.res.enkf.test_queue_config.QueueConfigTest)
addPythonTest(tests.res.enkf.test_log_config.LogConfigTest)
addPythonTest(tests.res.enkf.test_run_context.ErtRunContextTest)
addPythonTest(tests.res.enkf.test_enkf_runpath.EnKFRunpathTest)
addPythonTest(tests.res.enkf.test_programmatic_res_config.ProgrammaticResConfigTest)
addPythonTest(tests.res.enkf.test_enkf.EnKFTest)
addPythonTest(tests.res.enkf.test_enkf_transfer_env.EnKFTestTransferEnv)
addPythonTest(tests.res.enkf.test_simulation_batch.SimulationBatchTest)
addPythonTest(tests.res.enkf.test_model_config.ModelConfigTest)


if (STATOIL_TESTDATA_ROOT)
  addPythonTest(tests.res.enkf.test_enkf_obs.EnKFObsTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_enkf_load_results_manually.LoadResultsManuallyTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_enkf_fs.EnKFFSTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_enkf_fs_manager1.EnKFFSManagerTest1 LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_enkf_fs_manager2.EnKFFSManagerTest2 LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_enkf_library.EnKFLibraryTest)
  addPythonTest(tests.res.enkf.test_summary_key_set.SummaryKeySetTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_ecl_config.EclConfigTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_ert_context.ErtTestContextTest LABELS StatoilData)
  addPythonTest(tests.res.enkf.test_field_export.FieldExportTest LABELS StatoilData)

  # The following two tests involve loading the rml module as part of
  # the bootup process. That fails with a dlopen( ) error I just can
  # not understand; when the libraries have been installed with 'make
  # install' the tests seem to work.

  # addPythonTest(tests.res.enkf.test_labscale.LabScaleTest LABELS StatoilData)
  # addPythonTest(tests.res.enkf.test_update.UpdateTest LABELS StatoilData)
endif()

add_subdirectory(data)
add_subdirectory(export)
add_subdirectory(plot)
