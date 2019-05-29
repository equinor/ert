set(TEST_SOURCES
    __init__.py
    test_ert_run_context.py
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
    test_enkf_sim_model.py
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
    test_runpath_list_dump.py
    test_data_kw_define.py
    test_programmatic_res_config.py
    test_rng_config.py
    test_subst_config.py
    test_workflow_list.py
)

add_python_package("python.tests.res.enkf" ${PYTHON_INSTALL_PREFIX}/tests/res/enkf "${TEST_SOURCES}" False)


python_config_test(tests.res.enkf.test_data_kw_define.DataKWTest)
python_config_test(tests.res.enkf.test_es_update.ESUpdateTest)
python_config_test(tests.res.enkf.test_site_config.SiteConfigTest)
python_config_test(tests.res.enkf.test_res_config.ResConfigTest)
python_config_test(tests.res.enkf.test_forward_load_context.ForwardLoadContextTest)
python_config_test(tests.res.enkf.test_deprecation.DeprecationTest)
python_config_test(tests.res.enkf.test_meas_block.MeasBlockTest)
python_config_test(tests.res.enkf.test_meas_data.MeasDataTest)
python_config_test(tests.res.enkf.test_obs_data.ObsDataTest)
python_config_test(tests.res.enkf.test_gen_obs.GenObsTest)
python_config_test(tests.res.enkf.test_summary_obs.SummaryObsTest)
python_config_test(tests.res.enkf.test_block_obs.BlockObsTest)
python_config_test(tests.res.enkf.test_obs_block.ObsBlockTest)
python_config_test(tests.res.enkf.test_analysis_config.AnalysisConfigTest)
python_config_test(tests.res.enkf.test_analysis_iter_config.AnalysisIterConfigTest)
python_config_test(tests.res.enkf.test_state_map.StateMapTest)
python_config_test(tests.res.enkf.test_custom_kw_config_set.CustomKWConfigSetTest)
python_config_test(tests.res.enkf.test_summary_key_matcher.SummaryKeyMatcherTest)
python_config_test(tests.res.enkf.test_time_map.TimeMapTest)
python_config_test(tests.res.enkf.test_active_list.ActiveListTest)
python_config_test(tests.res.enkf.test_local_obsdata_node.LocalObsdataNodeTest)
python_config_test(tests.res.enkf.test_local_config.LocalConfigTest)
python_config_test(tests.res.enkf.test_ensemble_config.EnsembleConfigTest)
python_config_test(tests.res.enkf.test_hook_workflow.HookWorkFlowTest)
python_config_test(tests.res.enkf.test_runpath_list.RunpathListTest)
python_config_test(tests.res.enkf.test_field_config.FieldConfigTest)
python_config_test(tests.res.enkf.test_queue_config.QueueConfigTest)
python_config_test(tests.res.enkf.test_log_config.LogConfigTest)
python_config_test(tests.res.enkf.test_run_context.ErtRunContextTest)
python_config_test(tests.res.enkf.test_runpath_list_dump.RunpathListDumpTest)
python_config_test(tests.res.enkf.test_enkf_runpath.EnKFRunpathTest)
python_config_test(tests.res.enkf.test_programmatic_res_config.ProgrammaticResConfigTest)
python_config_test(tests.res.enkf.test_rng_config.RNGConfigTest)
python_config_test(tests.res.enkf.test_enkf.EnKFTest)
python_config_test(tests.res.enkf.test_enkf_transfer_env.EnKFTestTransferEnv)
python_config_test(tests.res.enkf.test_enkf_sim_model.EnKFTestSimModel)
python_config_test(tests.res.enkf.test_simulation_batch.SimulationBatchTest)
python_config_test(tests.res.enkf.test_model_config.ModelConfigTest)
python_config_test(tests.res.enkf.test_enkf_fs_manager1.EnKFFSManagerTest1)
python_config_test(tests.res.enkf.test_enkf_fs_manager2.EnKFFSManagerTest2)
python_config_test(tests.res.enkf.test_ert_run_context.ErtRunContextTest)



python_config_test(tests.res.enkf.test_enkf_obs.EnKFObsTest)
python_config_test(tests.res.enkf.test_enkf_load_results_manually.LoadResultsManuallyTest)
python_config_test(tests.res.enkf.test_enkf_fs.EnKFFSTest)
python_config_test(tests.res.enkf.test_enkf_library.EnKFLibraryTest)
python_config_test(tests.res.enkf.test_summary_key_set.SummaryKeySetTest)
python_config_test(tests.res.enkf.test_ecl_config.EclConfigTest)
python_config_test(tests.res.enkf.test_ert_context.ErtTestContextTest)
python_config_test(tests.res.enkf.test_field_export.FieldExportTest)
python_config_test(tests.res.enkf.test_subst_config.SubstConfigTest)
python_config_test(tests.res.enkf.test_workflow_list.ErtWorkflowListTest)

# The following two tests involve loading the rml module as part of
# the bootup process. That fails with a dlopen( ) error I just can
# not understand; when the libraries have been installed with 'make
# install' the tests seem to work.

# python_config_test(tests.res.enkf.test_labscale.LabScaleTest)
# python_config_test(tests.res.enkf.test_update.UpdateTest)

add_subdirectory(data)
add_subdirectory(export)
add_subdirectory(plot)
