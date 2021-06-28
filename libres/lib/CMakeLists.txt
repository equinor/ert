if(NOT SKBUILD)
  # Link directly to libecl if not using `pip install`
  find_package(ecl REQUIRED)
  set(ECL ecl)
  set(RES res)
endif()

add_library(res SHARED
  res_util/res_log.cpp
                res_util/log.cpp
                res_util/es_testdata.cpp
                res_util/arg_pack.cpp
                res_util/ui_return.cpp
                res_util/subst_list.cpp
                res_util/subst_func.cpp
                res_util/matrix_stat.cpp
                res_util/matrix_blas.cpp
                res_util/matrix_lapack.cpp
                res_util/matrix.cpp
                res_util/template.cpp
                res_util/path_fmt.cpp
                res_util/res_env.cpp
                res_util/res_portability.cpp
                res_util/util_printf.cpp
                res_util/block_fs.cpp
                res_util/res_version.cpp
                res_util/regression.cpp
                res_util/thread_pool.cpp
                res_util/template_loop.cpp  # Highly deprecated

                config/conf_util.cpp
                config/conf.cpp
                config/conf_data.cpp
                config/config_parser.cpp
                config/config_content.cpp
                config/config_path_stack.cpp
                config/config_content_item.cpp
                config/config_content_node.cpp
                config/config_error.cpp
                config/config_path_elm.cpp
                config/config_root_path.cpp
                config/config_schema_item.cpp
                config/config_settings.cpp

                rms/rms_file.cpp
                rms/rms_tag.cpp
                rms/rms_tagkey.cpp
                rms/rms_type.cpp
                rms/rms_util.cpp

                sched/history.cpp

                analysis/analysis_module.cpp
                analysis/bootstrap_enkf.cpp
                analysis/cv_enkf.cpp
                analysis/enkf_linalg.cpp
                analysis/fwd_step_enkf.cpp
                analysis/fwd_step_log.cpp
                analysis/module_data_block.cpp
                analysis/module_data_block_vector.cpp
                analysis/module_info.cpp
                analysis/module_obs_block.cpp
                analysis/module_obs_block_vector.cpp
                analysis/null_enkf.cpp
                analysis/sqrt_enkf.cpp
                analysis/std_enkf.cpp
                analysis/stepwise.cpp

                job_queue/ext_job.cpp
                job_queue/ext_joblist.cpp
                job_queue/forward_model.cpp
                job_queue/job_status.cpp
                job_queue/job_list.cpp
                job_queue/job_node.cpp
                job_queue/job_queue.cpp
                job_queue/job_queue_status.cpp
                job_queue/local_driver.cpp
                job_queue/lsf_driver.cpp
                job_queue/queue_driver.cpp
                job_queue/rsh_driver.cpp
                job_queue/slurm_driver.cpp
                job_queue/torque_driver.cpp
                job_queue/workflow.cpp
                job_queue/workflow_job.cpp
                job_queue/workflow_joblist.cpp
                job_queue/environment_varlist.cpp
                job_queue/job_kw_definitions.cpp
                ${lsb}

                enkf/active_list.cpp
                enkf/time_map.cpp
                enkf/analysis_config.cpp
                enkf/analysis_iter_config.cpp
                enkf/block_fs_driver.cpp
                enkf/block_obs.cpp
                enkf/callback_arg.cpp
                enkf/cases_config.cpp
                enkf/container.cpp
                enkf/container_config.cpp
                enkf/data_ranking.cpp
                enkf/ecl_config.cpp
                enkf/ecl_refcase_list.cpp
                enkf/enkf_analysis.cpp
                enkf/enkf_config_node.cpp
                enkf/enkf_defaults.cpp
                enkf/enkf_fs.cpp
                enkf/enkf_main.cpp
                enkf/enkf_main_ensemble.cpp
                enkf/enkf_main_jobs.cpp
                enkf/enkf_main_manage_fs.cpp
                enkf/enkf_node.cpp
                enkf/enkf_obs.cpp
                enkf/enkf_plot_data.cpp
                enkf/enkf_plot_gendata.cpp
                enkf/enkf_plot_gen_kw.cpp
                enkf/enkf_plot_gen_kw_vector.cpp
                enkf/enkf_plot_genvector.cpp
                enkf/enkf_plot_tvector.cpp
                enkf/enkf_serialize.cpp
                enkf/enkf_state.cpp
                enkf/enkf_types.cpp
                enkf/enkf_util.cpp
                enkf/ensemble_config.cpp
                enkf/ert_run_context.cpp
                enkf/ert_template.cpp
                enkf/ert_test_context.cpp
                enkf/ert_workflow_list.cpp
                enkf/ext_param.cpp
                enkf/ext_param_config.cpp
                enkf/field.cpp
                enkf/field_config.cpp
                enkf/field_trans.cpp
                enkf/forward_load_context.cpp
                enkf/fs_driver.cpp
                enkf/fs_types.cpp
                enkf/gen_common.cpp
                enkf/gen_data.cpp
                enkf/gen_data_config.cpp
                enkf/gen_kw.cpp
                enkf/gen_kw_config.cpp
                enkf/value_export.cpp
                enkf/gen_obs.cpp
                enkf/hook_manager.cpp
                enkf/hook_workflow.cpp
                enkf/local_config.cpp
                enkf/local_dataset.cpp
                enkf/local_ministep.cpp
                enkf/local_obsdata.cpp
                enkf/local_obsdata_node.cpp
                enkf/local_updatestep.cpp
                enkf/meas_data.cpp
                enkf/misfit_ensemble.cpp
                enkf/misfit_member.cpp
                enkf/misfit_ranking.cpp
                enkf/misfit_ts.cpp
                enkf/model_config.cpp
                enkf/obs_data.cpp
                enkf/obs_vector.cpp
                enkf/queue_config.cpp
                enkf/ranking_table.cpp
                enkf/rng_config.cpp
                enkf/run_arg.cpp
                enkf/runpath_list.cpp
                enkf/site_config.cpp
                enkf/rng_manager.cpp
                enkf/res_config.cpp
                enkf/row_scaling.cpp
                enkf/state_map.cpp
                enkf/state_map.cpp
                enkf/summary.cpp
                enkf/summary_config.cpp
                enkf/summary_key_matcher.cpp
                enkf/summary_key_set.cpp
                enkf/summary_obs.cpp
                enkf/surface.cpp
                enkf/surface_config.cpp
                enkf/trans_func.cpp
                enkf/subst_config.cpp
                enkf/log_config.cpp
                enkf/config_keys.cpp

                external/JSON/cJSON.cpp
)

#-----------------------------------------------------------------

add_library(rml_enkf SHARED
  analysis/modules/rml_enkf_config.cpp
  analysis/modules/rml_enkf.cpp
  analysis/modules/rml_enkf_common.cpp
  analysis/modules/rml_enkf_log.cpp)
target_link_libraries(rml_enkf PRIVATE ${RES} ${ECL} ${CMAKE_DL_LIBS})
target_include_directories(rml_enkf PRIVATE ${ECL_INCLUDE_DIRS} include analysis/modules)
set_target_properties(rml_enkf PROPERTIES VERSION 1.0
  SOVERSION 1.0
  PREFIX "")

add_library(ies SHARED
  analysis/modules/ies_enkf.cpp
  analysis/modules/ies_enkf_config.cpp
  analysis/modules/ies_enkf_data.cpp)
target_link_libraries(ies PRIVATE ${RES} ${ECL} ${CMAKE_DL_LIBS})
target_include_directories(ies PRIVATE ${ECL_INCLUDE_DIRS} include analysis/modules)

add_library(std_enkf_debug SHARED
  analysis/modules/std_enkf_debug.cpp)
target_link_libraries(std_enkf_debug PRIVATE ${RES} ${ECL} ${CMAKE_DL_LIBS})
target_include_directories(std_enkf_debug PRIVATE ${ECL_INCLUDE_DIRS} include modules)
set_target_properties(std_enkf_debug PROPERTIES VERSION 1.0
  SOVERSION 1.0
  PREFIX "")

#-----------------------------------------------------------------

# The INTERNAL_LINK property is required to get the
# system with analysis modules to work, might have some side-effects?
target_compile_definitions(res PRIVATE -DINTERNAL_LINK)

find_package(LAPACK REQUIRED)
target_link_libraries(res PUBLIC ${ECL} ${LAPACK_LIBRARIES} ${LAPACK_LINKER_FLAGS})
target_include_directories(res
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/private-include>
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/private-include/ext/json>
        PRIVATE "${ECL_INCLUDE_DIRS}")

target_compile_definitions(res PRIVATE
  -DGIT_COMMIT=${GIT_COMMIT}
  -DGIT_COMMIT_SHORT=${GIT_COMMIT_SHORT}
  -DRES_VERSION_MAJOR=${RES_VERSION_MAJOR}
  -DRES_VERSION_MINOR=${RES_VERSION_MINOR}
  -DRES_VERSION_MICRO=${RES_VERSION_MICRO}
  -DCOMPILE_TIME_STAMP="${RES_BUILD_TIME}")

install(TARGETS res
        EXPORT res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS rml_enkf
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS ies
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
