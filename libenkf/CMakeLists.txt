project(res-enkf C)

add_library(enkf src/active_list.c
                 src/time_map.c
                 src/analysis_config.c
                 src/analysis_iter_config.c
                 src/block_fs_driver.c
                 src/block_obs.c
                 src/cases_config.c
                 src/container.c
                 src/container_config.c
                 src/custom_kw.c
                 src/custom_kw_config.c
                 src/custom_kw_config_set.c
                 src/data_ranking.c
                 src/ecl_config.c
                 src/ecl_refcase_list.c
                 src/enkf_analysis.c
                 src/enkf_config_node.c
                 src/enkf_defaults.c
                 src/enkf_fs.c
                 src/enkf_main.c
                 src/enkf_main_jobs.c
                 src/enkf_node.c
                 src/enkf_obs.c
                 src/enkf_plot_data.c
                 src/enkf_plot_gendata.c
                 src/enkf_plot_gen_kw.c
                 src/enkf_plot_gen_kw_vector.c
                 src/enkf_plot_genvector.c
                 src/enkf_plot_tvector.c
                 src/enkf_serialize.c
                 src/enkf_state.c
                 src/enkf_types.c
                 src/enkf_util.c
                 src/ensemble_config.c
                 src/ert_run_context.c
                 src/ert_template.c
                 src/ert_test_context.c
                 src/ert_workflow_list.c
                 src/ext_param.c
                 src/ext_param_config.c
                 src/field.c
                 src/field_config.c
                 src/field_trans.c
                 src/forward_load_context.c
                 src/fs_driver.c
                 src/fs_types.c
                 src/gen_common.c
                 src/gen_data.c
                 src/gen_data_config.c
                 src/gen_kw.c
                 src/gen_kw_config.c
                 src/gen_obs.c
                 src/hook_manager.c
                 src/hook_workflow.c
                 src/local_config.c
                 src/local_context.c
                 src/local_dataset.c
                 src/local_ministep.c
                 src/local_obsdata.c
                 src/local_obsdata_node.c
                 src/local_updatestep.c
                 src/meas_data.c
                 src/member_config.c
                 src/misfit_ensemble.c
                 src/misfit_member.c
                 src/misfit_ranking.c
                 src/misfit_ts.c
                 src/model_config.c
                 src/obs_data.c
                 src/obs_vector.c
                 src/pca_plot_data.c
                 src/pca_plot_vector.c
                 src/plain_driver.c
                 src/plot_settings.c
                 src/queue_config.c
                 src/ranking_table.c
                 src/rng_config.c
                 src/run_arg.c
                 src/runpath_list.c
                 src/site_config.c
                 src/res_config.c
                 src/state_map.c
                 src/state_map.c
                 src/summary.c
                 src/summary_config.c
                 src/summary_key_matcher.c
                 src/summary_key_set.c
                 src/summary_obs.c
                 src/surface.c
                 src/surface_config.c
                 src/trans_func.c
                 src/subst_config.c
                 src/log_config.c
                 src/config_keys.c
)

target_include_directories(enkf
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)
target_link_libraries(enkf PUBLIC res_util sched analysis rms config job_queue ecl)
target_compile_definitions(enkf PRIVATE
    -DSITE_CONFIG_FILE=\"${SITE_CONFIG_FILE}\"
)

set_target_properties(enkf PROPERTIES
    VERSION ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR}
    SOVERSION ${RES_VERSION_MAJOR})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS enkf
        EXPORT  res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS)
    return ()
endif()

foreach (test   enkf_active_list
                enkf_analysis_config
                enkf_analysis_config_ext_module
                enkf_cases_config
                enkf_config_node
                enkf_enkf_config_node_gen_data
                enkf_config_node_ext_param
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
                enkf_obs_tests
                enkf_obs_vector
                enkf_pca_plot
                enkf_plot_data
                enkf_plot_gendata
                enkf_plot_gen_kw
                enkf_plot_gen_kw_vector
                enkf_plot_genvector
                enkf_plot_tvector
                enkf_run_arg
                enkf_state_map
                obs_vector_tests
                log_config_level_parse
    )
    add_executable(${test} tests/${test}.c)
    target_link_libraries(${test} enkf)
    add_test(NAME ${test} COMMAND ${test})
endforeach ()

function( add_config_test name command )
    add_test( NAME ${name}
              COMMAND ${command} ${ARGN})
     
    set_property( TEST ${name} PROPERTY ENVIRONMENT "ERT_ROOT=${ERT_ROOT}")
endfunction()

add_executable(enkf_queue_config tests/enkf_queue_config.c)
target_link_libraries(enkf_queue_config enkf)
add_config_test(enkf_queue_config
                enkf_queue_config)

add_executable(enkf_magic_string_in_workflows tests/enkf_magic_string_in_workflows.c)
target_link_libraries(enkf_magic_string_in_workflows enkf)
add_config_test(enkf_magic_string_in_workflows
                enkf_magic_string_in_workflows
                ${CMAKE_SOURCE_DIR}/test-data/local/config/workflows/config)

add_executable(enkf_main tests/enkf_main.c)
target_link_libraries(enkf_main enkf)
add_config_test(enkf_main
                enkf_main
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config rng)

add_executable(enkf_runpath_list tests/enkf_runpath_list.c)
target_link_libraries(enkf_runpath_list enkf)
add_config_test(enkf_runpath_list
                enkf_runpath_list
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/runpath_list/config)

add_executable(enkf_gen_obs_load tests/enkf_gen_obs_load.c)
target_link_libraries(enkf_gen_obs_load enkf)
add_config_test(enkf_gen_obs_load
                enkf_gen_obs_load
                ${CMAKE_SOURCE_DIR}/test-data/local/config/gen_data/config)

add_executable(enkf_ert_workflow_list tests/enkf_ert_workflow_list.c)
target_link_libraries(enkf_ert_workflow_list enkf)
add_config_test(enkf_ert_workflow_list
                enkf_ert_workflow_list
                ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/SCALE_STD)

add_executable(enkf_analysis_config_analysis_load tests/enkf_analysis_config_analysis_load.c)
target_link_libraries(enkf_analysis_config_analysis_load enkf)
add_config_test(enkf_analysis_config_analysis_load
                enkf_analysis_config_analysis_load
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/analysis_load_config)

# It seems necessary to set the LD_LIBRARY_PATH for the loading of the
# rml_enkf module to work.  Do understand why!
set_property(TEST enkf_analysis_config_analysis_load PROPERTY
    ENVIRONMENT
    "ERT_SITE_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/analysis_load_site_config;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")

add_executable(enkf_workflow_job_test_version tests/enkf_workflow_job_test_version.c)
target_link_libraries(enkf_workflow_job_test_version enkf)
add_config_test(enkf_workflow_job_test_version
                enkf_workflow_job_test_version
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/workflow_jobs)

add_executable(enkf_ert_test_context tests/enkf_ert_test_context.c)
target_link_libraries(enkf_ert_test_context enkf)
add_config_test(enkf_ert_test_context
                enkf_ert_test_context
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/test_context/config
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/test_context/wf_job
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/test_context/wf_job_fail)

add_executable(enkf_plot_gen_kw_fs tests/enkf_plot_gen_kw_fs.c)
target_link_libraries(enkf_plot_gen_kw_fs enkf )
add_config_test(enkf_plot_gen_kw_fs
                enkf_plot_gen_kw_fs
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/gen_kw_plot/config)

add_executable(gen_kw_test tests/gen_kw_test.c)
target_link_libraries(gen_kw_test enkf)
add_config_test(gen_kw_test
                gen_kw_test
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/forward/ert/config_GEN_KW_true)


add_executable(gen_kw_logarithmic_test tests/gen_kw_logarithmic_test.c)
target_link_libraries(gen_kw_logarithmic_test enkf)
add_config_test(gen_kw_logarithmic_test
                gen_kw_logarithmic_test
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/gen_kw_logarithmic/config_GEN_KW_logarithmic)

add_executable(enkf_forward_init_GEN_KW tests/enkf_forward_init_GEN_KW.c)
target_link_libraries(enkf_forward_init_GEN_KW enkf)
add_config_test(enkf_forward_init_GEN_KW_TRUE
                enkf_forward_init_GEN_KW
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/forward/ert
                config_GEN_KW_true
                TRUE)

add_config_test(enkf_forward_init_GEN_KW_FALSE
                enkf_forward_init_GEN_KW
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/forward/ert
                config_GEN_KW_false
                FALSE)


add_executable(enkf_select_case_job tests/enkf_select_case_job.c)
target_link_libraries(enkf_select_case_job enkf)
add_config_test(enkf_select_case_job
                enkf_select_case_job
                ${CMAKE_SOURCE_DIR}/test-data/local/snake_oil/snake_oil.ert
                ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal-tui/config/SELECT_CASE)


add_executable(enkf_forward_init_GEN_PARAM tests/enkf_forward_init_GEN_PARAM.c)
target_link_libraries(enkf_forward_init_GEN_PARAM enkf)

add_config_test(enkf_forward_init_GEN_PARAM_TRUE
                enkf_forward_init_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/forward/ert
                config_GEN_PARAM_true
                TRUE)

add_config_test(enkf_forward_init_GEN_PARAM_FALSE
                enkf_forward_init_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config/forward/ert
                config_GEN_PARAM_false
                FALSE)

add_executable(enkf_umask_config_test tests/enkf_umask_config_test.c)
target_link_libraries(enkf_umask_config_test enkf)
add_config_test(enkf_umask_config_test
                enkf_umask_config_test
                ${CMAKE_SOURCE_DIR}/test-data/local/simple_config/config_umask)

add_executable(enkf_rng tests/enkf_rng.c)
target_link_libraries(enkf_rng enkf)
add_config_test(enkf_rng
                enkf_rng ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config rng)

add_executable(enkf_forward_load_context tests/enkf_forward_load_context.c)
target_link_libraries(enkf_forward_load_context enkf)
add_config_test(enkf_forward_load_context
                enkf_forward_load_context
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/config
                forward_load_context)

add_executable(enkf_ensemble_GEN_PARAM tests/enkf_ensemble_GEN_PARAM.c)
target_link_libraries(enkf_ensemble_GEN_PARAM enkf)
add_config_test(enkf_ensemble_GEN_PARAM
                enkf_ensemble_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ensemble/GEN_PARAM)

if (NOT STATOIL_TESTDATA_ROOT)
    return ()
endif ()


function( add_statoil_test name command )
    add_test( NAME ${name}
              COMMAND ${command} ${ARGN})
     
    set_property( TEST ${name} PROPERTY LABELS StatoilData)
    set_property( TEST ${name} PROPERTY ENVIRONMENT "ERT_ROOT=${ERT_ROOT}")
endfunction()
              

add_executable(enkf_site_config tests/enkf_site_config.c)
target_link_libraries(enkf_site_config enkf)
add_statoil_test(enkf_site_config
                 enkf_site_config /project/res/etc/ERT/site-config)

add_executable(enkf_gen_data_config tests/enkf_gen_data_config.c)
target_link_libraries(enkf_gen_data_config enkf)
add_statoil_test(enkf_gen_data_config
                 enkf_gen_data_config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/gendata_test/RFT_E-3H_21
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/gendata_test/RFT_E-3H_21_empty)

add_executable(enkf_block_obs tests/enkf_block_obs.c)
target_link_libraries(enkf_block_obs enkf)
add_statoil_test(enkf_block_obs
                 enkf_block_obs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.EGRID)

add_executable(enkf_obs_fs tests/enkf_obs_fs.c)
target_link_libraries(enkf_obs_fs enkf)
add_statoil_test(enkf_obs_fs
                 enkf_obs_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/obs_testing/config)

add_executable(enkf_obs_vector_fs tests/enkf_obs_vector_fs.c)
target_link_libraries(enkf_obs_vector_fs enkf)
add_statoil_test(enkf_obs_vector_fs
                 enkf_obs_vector_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/obs_testing/config)

add_executable(enkf_plot_data_fs tests/enkf_plot_data_fs.c)
target_link_libraries(enkf_plot_data_fs enkf)
add_statoil_test(enkf_plot_data_fs
                 enkf_plot_data_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/plotData/config)

add_executable(enkf_time_map tests/enkf_time_map.c)
target_link_libraries(enkf_time_map enkf)
add_statoil_test(enkf_time_map1 enkf_time_map)
        add_statoil_test(enkf_time_map2
                 enkf_time_map
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/ModifiedSummary/EXTRA_TSTEP
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/ModifiedSummary/SHORT
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/ModifiedSummary/MISSING_TSTEP)

add_executable(enkf_main_fs tests/enkf_main_fs.c)
target_link_libraries(enkf_main_fs enkf)
add_statoil_test(enkf_main_fs
                 enkf_main_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/plotData/config)

add_executable(enkf_main_fs_current_file_test tests/enkf_main_fs_current_file_test.c)
target_link_libraries(enkf_main_fs_current_file_test enkf)
add_statoil_test(enkf_main_fs_current_file_test
                 enkf_main_fs_current_file_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/plotData/config)

add_executable(enkf_scale_correlated_std tests/enkf_scale_correlated_std.c)
target_link_libraries(enkf_scale_correlated_std enkf)
add_statoil_test(enkf_scale_correlated_std
                 enkf_scale_correlated_std
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/with_data/config
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/STD_SCALE_CORRELATED_OBS)

add_executable(enkf_plot_gendata_fs tests/enkf_plot_gendata_fs.c)
target_link_libraries(enkf_plot_gendata_fs  enkf)
add_statoil_test(enkf_plot_gendata_fs
                 enkf_plot_gendata_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/with_GEN_DATA/config)

add_executable(enkf_state_report_step_compatible
               tests/enkf_state_report_step_compatible.c)
target_link_libraries(enkf_state_report_step_compatible  enkf)
add_statoil_test(enkf_state_report_step_compatible_TRUE
                 enkf_state_report_step_compatible
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_sum_compatible_true
                 config_ecl_sum_compatible_true
                 TRUE)

add_statoil_test(enkf_state_report_step_compatible_FALSE
                 enkf_state_report_step_compatible
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_sum_compatible_false
                 config_ecl_sum_compatible_false
                 FALSE)

add_executable(enkf_state_manual_load_test tests/enkf_state_manual_load_test.c)
target_link_libraries(enkf_state_manual_load_test enkf)
add_statoil_test(enkf_state_manual_load_test
                 enkf_state_manual_load_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_sum_compatible_true
                 config_ecl_sum_compatible_true)

add_executable(enkf_state_skip_summary_load_test tests/enkf_state_skip_summary_load_test.c)
target_link_libraries(enkf_state_skip_summary_load_test enkf)

add_statoil_test(enkf_state_summary_vars_present
                 enkf_state_skip_summary_load_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_summary_vars_config
                 config_summary_vars)

add_statoil_test(enkf_state_no_summary_vars_present
                 enkf_state_skip_summary_load_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_no_summary_vars_config
                 config_no_summary_vars)

add_executable(enkf_export_field_test tests/enkf_export_field_test.c)
target_link_libraries(enkf_export_field_test enkf)
add_statoil_test(enkf_export_field_test
                 enkf_export_field_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/export_fields/config
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_FIELD
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_FIELD_ECL_GRDECL
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_FIELD_RMS_ROFF)

add_executable(enkf_workflow_job_test tests/enkf_workflow_job_test.c)
target_link_libraries(enkf_workflow_job_test enkf)
add_statoil_test(enkf_workflow_job_test
                 enkf_workflow_job_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/with_data/config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/enkf_state_runpath/config_runpath_multiple_iterations
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal-tui/config/CREATE_CASE
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal-tui/config/INIT_CASE_FROM_EXISTING
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/LOAD_RESULTS
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/LOAD_RESULTS_ITER
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/OBSERVATION_RANKING
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/DATA_RANKING
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_RANKING
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/INIT_MISFIT_TABLE
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_RUNPATH
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/PRE_SIMULATION_COPY)

add_executable(enkf_forward_init_SURFACE tests/enkf_forward_init_SURFACE.c)
target_link_libraries(enkf_forward_init_SURFACE enkf)
add_statoil_test(enkf_forward_init_SURFACE_TRUE
                 enkf_forward_init_SURFACE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/surface config_surface_true
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/surface/Surface.irap
                 TRUE)

add_statoil_test(enkf_forward_init_SURFACE_FALSE
                 enkf_forward_init_SURFACE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/surface config_surface_false
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/surface/Surface.irap
                 FALSE)

add_executable(enkf_forward_init_FIELD tests/enkf_forward_init_FIELD.c)
target_link_libraries(enkf_forward_init_FIELD enkf)
add_statoil_test(enkf_forward_init_FIELD_TRUE
                 enkf_forward_init_FIELD
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/field config_field_true
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/field/petro.grdecl
                 TRUE)

add_statoil_test(enkf_forward_init_FIELD_FALSE
                 enkf_forward_init_FIELD
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/field config_field_false
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/forward_init/field/petro.grdecl
                 FALSE)

add_executable(enkf_forward_init_transform tests/enkf_forward_init_transform.c)
target_link_libraries(enkf_forward_init_transform enkf)
add_statoil_test(enkf_forward_init_transform_TRUE
                 enkf_forward_init_transform
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/transform transform_forward_init_true
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/transform/petro.grdecl
                 TRUE)

add_statoil_test(enkf_forward_init_transform_FALSE
                 enkf_forward_init_transform
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/transform transform_forward_init_false
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/transform/petro.grdecl
                 FALSE)

add_executable(enkf_export_inactive_cells tests/enkf_export_inactive_cells.c)
target_link_libraries(enkf_export_inactive_cells enkf)
add_statoil_test(enkf_export_inactive_cells
                 enkf_export_inactive_cells
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/export_inactive_cells/config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/export_inactive_cells/petro.grdecl)

add_executable(enkf_refcase_list tests/enkf_refcase_list.c)
target_link_libraries(enkf_refcase_list enkf)
add_statoil_test(enkf_refcase_list
                 enkf_refcase_list
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat*/ECLIPSE)

add_statoil_test(enkf_refcase_list2
                 enkf_refcase_list
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat*/ECLIPSE.*)

add_executable(enkf_ecl_config tests/enkf_ecl_config.c)
target_link_libraries(enkf_ecl_config enkf)
add_statoil_test(enkf_ecl_config1 enkf_ecl_config)
        add_statoil_test(enkf_ecl_config2
                 enkf_ecl_config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE)

add_executable(enkf_ecl_config_config tests/enkf_ecl_config_config.c)
target_link_libraries(enkf_ecl_config_config enkf)
add_statoil_test(enkf_ecl_config_config
                 enkf_ecl_config_config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_config)
