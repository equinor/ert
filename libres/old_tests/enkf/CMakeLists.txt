# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------

function(add_config_test name command)
  add_test(NAME ${name} COMMAND ${command} ${ARGN})

  set_property(TEST ${name} PROPERTY ENVIRONMENT
                                     "ERT_SITE_CONFIG=${SITE_CONFIG_PATH}")
endfunction()

function(add_equinor_test name command)
  add_test(NAME ${name} COMMAND ${command} ${ARGN})

  set_property(TEST ${name} PROPERTY LABELS EquinorData)
endfunction()

# -----------------------------------------------------------------
# Public tests
# -----------------------------------------------------------------

foreach(
  test
  enkf_active_list
  enkf_analysis_config
  enkf_config_node
  enkf_enkf_config_node_gen_data
  enkf_ensemble
  enkf_ensemble_config
  enkf_ert_run_context
  enkf_fs
  enkf_gen_data_config_parse
  enkf_iter_config
  enkf_local_obsdata
  enkf_local_obsdata_node
  enkf_meas_data
  enkf_model_config
  enkf_obs_invalid_path
  enkf_obs
  enkf_obs_vector
  enkf_plot_data
  enkf_plot_gendata
  enkf_plot_gen_kw
  enkf_plot_gen_kw_vector
  enkf_plot_genvector
  enkf_plot_tvector
  enkf_update_log
  enkf_run_arg
  enkf_state_map
  obs_vector
  measured
  obs_mask_deactivate
  row_scaling
  rng_manager
  trans_func
  value_export)
  add_executable(${test} test_${test}.cpp)
  target_link_libraries(${test} res)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

# rng_config test is unique in that it requires mocking of logs
add_executable(rng_config test_rng_config.cpp ../../tests/logging.cpp)
target_link_libraries(rng_config res)
add_test(NAME rng_config COMMAND rng_config)

foreach(
  test
  enkf_magic_string_in_workflows
  enkf_queue_config
  enkf_gen_obs_load
  enkf_ert_workflow_list
  enkf_main
  enkf_select_case_job
  enkf_plot_gen_kw_fs
  enkf_workflow_job_test_directory
  enkf_ert_test_context
  gen_kw_logarithmic
  enkf_forward_init_GEN_KW
  enkf_forward_init_GEN_PARAM
  enkf_umask_config
  enkf_executable_path
  enkf_forward_load_context
  enkf_ensemble_GEN_PARAM
  gen_kw
  enkf_runpath_list)

  add_executable(${test} test_${test}.cpp)
  target_link_libraries(${test} res)
endforeach()

add_config_test(enkf_executable_path enkf_executable_path)

add_config_test(enkf_forward_load_context enkf_forward_load_context
                "${TEST_DATA_DIR}/enkf/config" forward_load_context)

add_config_test(enkf_ensemble_GEN_PARAM enkf_ensemble_GEN_PARAM
                "${TEST_DATA_DIR}/enkf/ensemble/GEN_PARAM")

add_config_test(enkf_umask_config enkf_umask_config
                "${LOCAL_TEST_DATA_DIR}/simple_config/config_umask")

add_config_test(
  enkf_forward_init_GEN_PARAM_TRUE enkf_forward_init_GEN_PARAM
  "${TEST_DATA_DIR}/enkf/config/forward/ert" config_GEN_PARAM_true TRUE)

add_config_test(
  enkf_forward_init_GEN_PARAM_FALSE enkf_forward_init_GEN_PARAM
  "${TEST_DATA_DIR}/enkf/config/forward/ert" config_GEN_PARAM_false FALSE)

add_config_test(
  enkf_select_case_job enkf_select_case_job
  "${LOCAL_TEST_DATA_DIR}/snake_oil/snake_oil.ert"
  "${SHARE_DIR}/workflows/jobs/internal-tui/config/SELECT_CASE")
add_config_test(
  enkf_forward_init_GEN_KW_TRUE enkf_forward_init_GEN_KW
  "${TEST_DATA_DIR}/enkf/config/forward/ert" config_GEN_KW_true TRUE)
add_config_test(
  enkf_forward_init_GEN_KW_FALSE enkf_forward_init_GEN_KW
  "${TEST_DATA_DIR}/enkf/config/forward/ert" config_GEN_KW_false FALSE)
add_config_test(enkf_queue_config enkf_queue_config)
add_config_test(enkf_magic_string_in_workflows enkf_magic_string_in_workflows
                "${LOCAL_TEST_DATA_DIR}/config/workflows/config")
add_config_test(enkf_main enkf_main "${TEST_DATA_DIR}/enkf/config" rng)
add_config_test(enkf_runpath_list enkf_runpath_list
                "${TEST_DATA_DIR}/enkf/config/runpath_list/config")
add_config_test(enkf_gen_obs_load enkf_gen_obs_load
                "${LOCAL_TEST_DATA_DIR}/poly_example/poly.ert")
add_config_test(
  enkf_ert_workflow_list enkf_ert_workflow_list
  ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/SCALE_STD)
add_config_test(
  enkf_ert_test_context
  enkf_ert_test_context
  "${TEST_DATA_DIR}/enkf/config/test_context/config"
  "${TEST_DATA_DIR}/enkf/config/test_context/wf_job"
  "${TEST_DATA_DIR}/enkf/config/test_context/wf_job_fail")
add_config_test(enkf_plot_gen_kw_fs enkf_plot_gen_kw_fs
                "${TEST_DATA_DIR}/enkf/config/gen_kw_plot/config")
add_config_test(gen_kw gen_kw
                "${TEST_DATA_DIR}/enkf/config/forward/ert/config_GEN_KW_true")
add_config_test(
  gen_kw_logarithmic gen_kw_logarithmic
  "${TEST_DATA_DIR}/enkf/config/gen_kw_logarithmic/config_GEN_KW_logarithmic")

add_config_test(
  enkf_workflow_job_test_directory enkf_workflow_job_test_directory
  "${TEST_DATA_DIR}/enkf/workflow_jobs")

foreach(
  test
  enkf_site_config
  enkf_main_fs_current_file
  enkf_plot_gendata_fs
  enkf_state_report_step_compatible
  enkf_state_manual_load
  enkf_export_field
  enkf_workflow_job
  enkf_forward_init_SURFACE
  enkf_forward_init_FIELD
  enkf_forward_init_transform
  enkf_state_skip_summary_load
  enkf_export_inactive_cells
  enkf_refcase_list
  enkf_ecl_config
  enkf_ecl_config_config
  enkf_gen_data_config
  enkf_block_obs
  enkf_obs_fs
  enkf_obs_vector_fs
  enkf_plot_data_fs
  enkf_time_map
  enkf_main_fs)

  add_executable(${test} test_${test}.cpp)
  target_link_libraries(${test} res)
endforeach()

# -----------------------------------------------------------------
# Equinor tests
# -----------------------------------------------------------------

if(NOT EXISTS "${EQUINOR_TEST_DATA_DIR}")
  return()
endif()

add_equinor_test(enkf_site_config enkf_site_config
                 /project/res/etc/ERT/site-config)

add_equinor_test(
  enkf_gen_data_config enkf_gen_data_config
  "${EQUINOR_TEST_DATA_DIR}/config/gendata_test/RFT_E-3H_21"
  "${EQUINOR_TEST_DATA_DIR}/config/gendata_test/RFT_E-3H_21_empty")

add_equinor_test(enkf_block_obs enkf_block_obs
                 "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE.EGRID")

add_equinor_test(enkf_obs_fs enkf_obs_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/obs_testing/config")

add_equinor_test(enkf_obs_vector_fs enkf_obs_vector_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/obs_testing/config")

add_equinor_test(enkf_plot_data_fs enkf_plot_data_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/plotData/config")

add_equinor_test(enkf_time_map1 enkf_time_map)
add_equinor_test(
  enkf_time_map2
  enkf_time_map
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/EXTRA_TSTEP"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/SHORT"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/ModifiedSummary/MISSING_TSTEP")

add_equinor_test(enkf_main_fs enkf_main_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/plotData/config")

add_equinor_test(enkf_main_fs_current_file enkf_main_fs_current_file
                 "${EQUINOR_TEST_DATA_DIR}/config/plotData/config")

add_equinor_test(enkf_plot_gendata_fs enkf_plot_gendata_fs
                 "${EQUINOR_TEST_DATA_DIR}/config/with_GEN_DATA/config")

add_equinor_test(
  enkf_state_report_step_compatible_TRUE enkf_state_report_step_compatible
  "${EQUINOR_TEST_DATA_DIR}/config/ecl_sum_compatible_true"
  config_ecl_sum_compatible_true TRUE)

add_equinor_test(
  enkf_state_manual_load enkf_state_manual_load
  "${EQUINOR_TEST_DATA_DIR}/config/ecl_sum_compatible_true"
  config_ecl_sum_compatible_true)

add_equinor_test(
  enkf_state_summary_vars_present enkf_state_skip_summary_load
  "${EQUINOR_TEST_DATA_DIR}/config/ecl_summary_vars_config" config_summary_vars)

add_equinor_test(
  enkf_state_no_summary_vars_present enkf_state_skip_summary_load
  "${EQUINOR_TEST_DATA_DIR}/config/ecl_no_summary_vars_config"
  config_no_summary_vars)

add_equinor_test(
  enkf_export_field
  enkf_export_field
  "${EQUINOR_TEST_DATA_DIR}/config/export_fields/config"
  "${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_FIELD"
  "${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_FIELD_ECL_GRDECL"
  "${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_FIELD_RMS_ROFF")

add_equinor_test(
  enkf_workflow_job
  enkf_workflow_job
  "${EQUINOR_TEST_DATA_DIR}/config/with_data/config"
  "${EQUINOR_TEST_DATA_DIR}/config/enkf_state_runpath/config_runpath_multiple_iterations"
  "${SHARE_DIR}/workflows/jobs/internal-tui/config/CREATE_CASE"
  "${SHARE_DIR}/workflows/jobs/internal-tui/config/INIT_CASE_FROM_EXISTING"
  "${SHARE_DIR}/workflows/jobs/internal/config/INIT_MISFIT_TABLE"
  "${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_RUNPATH"
  "${SHARE_DIR}/workflows/jobs/internal/config/PRE_SIMULATION_COPY")

add_equinor_test(
  enkf_forward_init_SURFACE_TRUE enkf_forward_init_SURFACE
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/surface" config_surface_true
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/surface/Surface.irap" TRUE)

add_equinor_test(
  enkf_forward_init_SURFACE_FALSE enkf_forward_init_SURFACE
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/surface" config_surface_false
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/surface/Surface.irap" FALSE)

add_equinor_test(
  enkf_forward_init_FIELD_TRUE enkf_forward_init_FIELD
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/field" config_field_true
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/field/petro.grdecl" TRUE)

add_equinor_test(
  enkf_forward_init_FIELD_FALSE enkf_forward_init_FIELD
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/field" config_field_false
  "${EQUINOR_TEST_DATA_DIR}/config/forward_init/field/petro.grdecl" FALSE)

add_equinor_test(
  enkf_forward_init_transform_TRUE enkf_forward_init_transform
  "${EQUINOR_TEST_DATA_DIR}/config/transform" transform_forward_init_true
  "${EQUINOR_TEST_DATA_DIR}/config/transform/petro.grdecl" TRUE)

add_equinor_test(
  enkf_forward_init_transform_FALSE enkf_forward_init_transform
  "${EQUINOR_TEST_DATA_DIR}/config/transform" transform_forward_init_false
  "${EQUINOR_TEST_DATA_DIR}/config/transform/petro.grdecl" FALSE)

add_equinor_test(
  enkf_export_inactive_cells enkf_export_inactive_cells
  "${EQUINOR_TEST_DATA_DIR}/config/export_inactive_cells/config"
  "${EQUINOR_TEST_DATA_DIR}/config/export_inactive_cells/petro.grdecl")

add_equinor_test(
  enkf_refcase_list enkf_refcase_list
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat*/ECLIPSE")

add_equinor_test(
  enkf_refcase_list2 enkf_refcase_list
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE"
  "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat*/ECLIPSE.*")

add_equinor_test(enkf_ecl_config1 enkf_ecl_config)
add_equinor_test(enkf_ecl_config2 enkf_ecl_config
                 "${EQUINOR_TEST_DATA_DIR}/ECLIPSE/Gurbat/ECLIPSE")

add_equinor_test(enkf_ecl_config_config enkf_ecl_config_config
                 "${EQUINOR_TEST_DATA_DIR}/config/ecl_config")
