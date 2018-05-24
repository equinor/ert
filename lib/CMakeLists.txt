if (LSF_LIBRARY AND LSF_HEADER_PATH)
    set(lsb job_queue/lsb.c)
    install(DIRECTORY include/
            DESTINATION include
            PATTERN lsb.h)
    target_compile_definitions(job_queue PUBLIC -DHAVE_LSF_LIBRARY)
    set(lsf ${LSF_LIBRARY})
else ()
   message(STATUS "LSF not found")
endif ()

add_library(res res_util/res_log.c
                res_util/log.c
                res_util/ui_return.c
                res_util/subst_list.c
                res_util/subst_func.c
                res_util/matrix_stat.c
                res_util/matrix_blas.c
                res_util/matrix_lapack.c
                res_util/matrix.c
                res_util/thread_pool.c
                res_util/template.c
                res_util/template_loop.c
                res_util/path_fmt.c
                res_util/res_env.c
                res_util/res_portability.c
                res_util/block_fs.c
                res_util/res_version.c
                res_util/regression.c
                res_util/thread_pool.c
                res_util/template_loop.c  # Highly deprecated
                res_util/block_fs.c

                config/conf_util.c
                config/conf.c
                config/conf_data.c
                config/config_parser.c
                config/config_content.c
                config/config_path_stack.c
                config/config_content_item.c
                config/config_content_node.c
                config/config_error.c
                config/config_path_elm.c
                config/config_root_path.c
                config/config_schema_item.c
                config/config_settings.c

                rms/rms_export.c
                rms/rms_file.c
                rms/rms_stats.c
                rms/rms_tag.c
                rms/rms_tagkey.c
                rms/rms_type.c
                rms/rms_util.c

                sched/group_history.cpp
                sched/group_index.cpp
                sched/gruptree.cpp
                sched/history.cpp
                sched/sched_blob.cpp
                sched/sched_file.cpp
                sched/sched_history.cpp
                sched/sched_kw.cpp
                sched/sched_kw_compdat.cpp
                sched/sched_kw_dates.cpp
                sched/sched_kw_gruptree.cpp
                sched/sched_kw_include.cpp
                sched/sched_kw_tstep.cpp
                sched/sched_kw_untyped.cpp
                sched/sched_kw_wconhist.cpp
                sched/sched_kw_wconinj.cpp
                sched/sched_kw_wconinje.cpp
                sched/sched_kw_wconinjh.cpp
                sched/sched_kw_wconprod.cpp
                sched/sched_kw_welspecs.cpp
                sched/sched_time.cpp
                sched/sched_types.cpp
                sched/sched_util.cpp
                sched/well_history.cpp
                sched/well_index.cpp

                analysis/analysis_module.c
                analysis/bootstrap_enkf.c
                analysis/cv_enkf.c
                analysis/enkf_linalg.c
                analysis/fwd_step_enkf.c
                analysis/fwd_step_log.c
                analysis/module_data_block.c
                analysis/module_data_block_vector.c
                analysis/module_info.c
                analysis/module_obs_block.c
                analysis/module_obs_block_vector.c
                analysis/null_enkf.c
                analysis/sqrt_enkf.c
                analysis/std_enkf.c
                analysis/stepwise.c

                job_queue/ext_job.c
                job_queue/ext_joblist.c
                job_queue/forward_model.c
                job_queue/job_status.c
                job_queue/job_list.c
                job_queue/job_node.c
                job_queue/job_queue.c
                job_queue/job_queue_manager.c
                job_queue/job_queue_status.c
                job_queue/local_driver.c
                job_queue/lsf_driver.c
                job_queue/queue_driver.c
                job_queue/rsh_driver.c
                job_queue/torque_driver.c
                job_queue/workflow.c
                job_queue/workflow_job.c
                job_queue/workflow_joblist.c
                job_queue/environment_varlist.c
                job_queue/job_kw_definitions.c
                ${lsb}

                enkf/active_list.c
                enkf/time_map.c
                enkf/analysis_config.c
                enkf/analysis_iter_config.c
                enkf/block_fs_driver.c
                enkf/block_obs.c
                enkf/callback_arg.c
                enkf/cases_config.c
                enkf/container.c
                enkf/container_config.c
                enkf/custom_kw.c
                enkf/custom_kw_config.c
                enkf/custom_kw_config_set.c
                enkf/data_ranking.c
                enkf/ecl_config.c
                enkf/ecl_refcase_list.c
                enkf/enkf_analysis.c
                enkf/enkf_config_node.c
                enkf/enkf_defaults.c
                enkf/enkf_fs.c
                enkf/enkf_main.c
                enkf/enkf_main_jobs.c
                enkf/enkf_node.c
                enkf/enkf_obs.c
                enkf/enkf_plot_data.c
                enkf/enkf_plot_gendata.c
                enkf/enkf_plot_gen_kw.c
                enkf/enkf_plot_gen_kw_vector.c
                enkf/enkf_plot_genvector.c
                enkf/enkf_plot_tvector.c
                enkf/enkf_serialize.c
                enkf/enkf_state.c
                enkf/enkf_types.c
                enkf/enkf_util.c
                enkf/ensemble_config.c
                enkf/ert_run_context.c
                enkf/ert_template.c
                enkf/ert_test_context.c
                enkf/ert_workflow_list.c
                enkf/ext_param.c
                enkf/ext_param_config.c
                enkf/field.c
                enkf/field_config.c
                enkf/field_trans.c
                enkf/forward_load_context.c
                enkf/fs_driver.c
                enkf/fs_types.c
                enkf/gen_common.c
                enkf/gen_data.c
                enkf/gen_data_config.c
                enkf/gen_kw.c
                enkf/gen_kw_config.c
                enkf/value_export.c
                enkf/gen_obs.c
                enkf/hook_manager.c
                enkf/hook_workflow.c
                enkf/local_config.c
                enkf/local_context.c
                enkf/local_dataset.c
                enkf/local_ministep.c
                enkf/local_obsdata.c
                enkf/local_obsdata_node.c
                enkf/local_updatestep.c
                enkf/meas_data.c
                enkf/misfit_ensemble.c
                enkf/misfit_member.c
                enkf/misfit_ranking.c
                enkf/misfit_ts.c
                enkf/model_config.c
                enkf/obs_data.c
                enkf/obs_vector.c
                enkf/pca_plot_data.c
                enkf/pca_plot_vector.c
                enkf/plain_driver.c
                enkf/plot_settings.c
                enkf/queue_config.c
                enkf/ranking_table.c
                enkf/rng_config.c
                enkf/run_arg.c
                enkf/runpath_list.c
                enkf/site_config.c
                enkf/rng_manager.c
                enkf/res_config.c
                enkf/state_map.c
                enkf/state_map.c
                enkf/summary.c
                enkf/summary_config.c
                enkf/summary_key_matcher.c
                enkf/summary_key_set.c
                enkf/summary_obs.c
                enkf/surface.c
                enkf/surface_config.c
                enkf/trans_func.c
                enkf/subst_config.c
                enkf/log_config.c
                enkf/config_keys.c
)

#-----------------------------------------------------------------

add_library(rml_enkf SHARED analysis/modules/rml_enkf_config.c
                            analysis/modules/rml_enkf.c
                            analysis/modules/rml_enkf_common.c
                            analysis/modules/rml_enkf_log.c
)
target_link_libraries(rml_enkf PRIVATE res ecl ${CMAKE_DL_LIBS})
target_include_directories(rml_enkf PRIVATE analysis/modules)
set_target_properties(rml_enkf PROPERTIES VERSION 1.0
                                          SOVERSION 1.0
                                          PREFIX "")

add_library(std_enkf_debug SHARED analysis/modules/std_enkf_debug.c)
target_link_libraries(std_enkf_debug PRIVATE res ecl ${CMAKE_DL_LIBS})
target_include_directories(std_enkf_debug PRIVATE modules)
set_target_properties(std_enkf_debug PROPERTIES VERSION 1.0
                                                SOVERSION 1.0
                                                PREFIX "")

#-----------------------------------------------------------------

# The INTERNAL_LINK property is required to get the
# system with analysis modules to work, might have some side-effects?
target_compile_definitions(res PRIVATE -DINTERNAL_LINK)

find_package(LAPACK REQUIRED)
target_link_libraries( res PUBLIC ecl ${LAPACK_LIBRARIES} ${LAPACK_LINKER_FLAGS})
target_include_directories(res
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )


set_target_properties( res PROPERTIES VERSION ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR} SOVERSION ${RES_VERSION_MAJOR} )

target_compile_definitions(res PRIVATE
            -DGIT_COMMIT=${GIT_COMMIT}
            -DGIT_COMMIT_SHORT=${GIT_COMMIT_SHORT}
            -DRES_VERSION_MAJOR=${RES_VERSION_MAJOR}
            -DRES_VERSION_MINOR=${RES_VERSION_MINOR}
            -DRES_VERSION_MICRO=${RES_VERSION_MICRO}
            -DCOMPILE_TIME_STAMP="${RES_BUILD_TIME}"
            -DSITE_CONFIG_FILE=\"${SITE_CONFIG_FILE}\"
)





install(DIRECTORY include/ DESTINATION include)
install(TARGETS res
        EXPORT res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#-----------------------------------------------------------------

if (NOT BUILD_TESTS)
    return ()
endif()

foreach(name ert_util_logh
             ert_util_subst_list
             ert_util_block_fs
             test_thread_pool
             res_util_PATH)

       add_executable(${name} res_util/tests/${name}.c)
       target_link_libraries(${name} res)
       add_test(NAME ${name} COMMAND ${name})
endforeach()

find_library( VALGRIND NAMES valgr )
if (VALGRIND)
    set(valgrind_cmd valgrind --error-exitcode=1 --tool=memcheck)
    add_test(NAME test_thread_pool COMMAND ${valgrind_cmd} test_thread_pool)
endif()

#-----------------------------------------------------------------

foreach(name config_content_node
             config_path_elm
             config_error
             config_content
             config_config
             config_schema_item)

    add_executable(${name} config/tests/${name}.c)
    target_link_libraries(${name} res)
    add_test(NAME ${name} COMMAND ${name})
endforeach()


foreach(name config_append_test
             config_node_test
             config_typeOK
             config_typeFail
             config_root_path
             config_include_test
             config_content_item
             config_define
             config_argc)

    add_executable(${name} config/tests/${name}.c)
    target_link_libraries(${name} res)
endforeach()

macro(config_test name input_file)
   add_test(NAME ${name} COMMAND ${name} ${CMAKE_CURRENT_SOURCE_DIR}/config/tests/data/${input_file})
endmacro()

config_test( config_append_test append_test)
config_test( config_node_test append_test)
config_test( config_typeOK type_testOK)
config_test( config_typeFail type_testFail)
config_test( config_content_item content_item_test)
config_test( config_define define_test)


add_test(NAME config_root_path
         COMMAND config_root_path ${CMAKE_CURRENT_SOURCE_DIR}/config/tests/data)

add_test(NAME config_include_test
         COMMAND config_include_test ${CMAKE_CURRENT_SOURCE_DIR}/config/tests/data include_test)

add_test(NAME config_argc
         COMMAND config_argc
            ${CMAKE_CURRENT_SOURCE_DIR}/config/tests/data/argc_OK
            ${CMAKE_CURRENT_SOURCE_DIR}/config/tests/data/argc_less
            ${CMAKE_CURRENT_SOURCE_DIR}/config/tests/data/argc_more)



#-----------------------------------------------------------------

add_executable(sched_history_summary sched/tests/sched_history_summary.c)
target_link_libraries(sched_history_summary res)

add_executable(rms_file_test rms/tests/rms_file_test.c)
target_link_libraries(rms_file_test res)

add_executable(analysis_external_module analysis/tests/analysis_test_external_module.c)
target_link_libraries(analysis_external_module res)

add_test(NAME analysis_module_rml
         COMMAND analysis_external_module
                 "RML_ENKF"
                 $<TARGET_FILE:rml_enkf> 41
                 ITER:45
                 USE_PRIOR:False
                 LAMBDA_REDUCE:0.10
                 LAMBDA_INCREASE:2.5
                 ENKF_TRUNCATION:0.77
                 LAMBDA0:0.25
                 LAMBDA_MIN:0.01
                 LOG_FILE:LogFile.txt
                 CLEAR_LOG:True
                 LAMBDA_RECALCULATE:True)

add_executable(analysis_test_module_info analysis/tests/analysis_test_module_info.c)
target_link_libraries(analysis_test_module_info res)
add_test(NAME analysis_test_module_info COMMAND analysis_test_module_info)

#-----------------------------------------------------------------


foreach(name job_status_test
             job_node_test
             job_lsf_parse_bsub_stdout
             job_lsf_test
             job_queue_driver_test
             job_torque_test
             job_queue_manager)

    add_executable(${name} job_queue/tests/${name}.c)
    target_link_libraries(${name} res)
    add_test(NAME ${name} COMMAND ${name})
endforeach()

# job_workflow requires this binary being available
add_executable(create_file job_queue/tests/create_file.c)
target_link_libraries(create_file res)

add_executable(job_loadOK job_queue/tests/job_loadOK.c)
target_link_libraries(job_loadOK res)
add_test(NAME job_loadOK1
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/internalOK)
add_test(NAME job_loadOK2
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/externalOK)
add_test(NAME job_loadOK3
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/internalOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/externalOK)

add_executable(job_loadFail job_queue/tests/job_loadFail.c)
target_link_libraries(job_loadFail res)

add_test(NAME job_loadFail1
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/internalFail)
add_test(NAME job_loadFail2
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/externalFail)
add_test(NAME job_loadFail3
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/internalFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/externalFail)

add_executable(job_list_test job_queue/tests/job_list_test.c)
target_link_libraries(job_list_test res)
add_test(NAME job_list_test
         COMMAND valgrind --leak-check=full --error-exitcode=1
                 $<TARGET_FILE:job_list_test>)

add_executable(job_workflow_test job_queue/tests/job_workflow_test.c)
target_link_libraries(job_workflow_test res)
add_test(NAME job_workflow_test
         COMMAND job_workflow_test
                 ${EXECUTABLE_OUTPUT_PATH}
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/internal_job)

add_executable(job_lsf_exclude_hosts_test job_queue/tests/job_lsf_exclude_hosts_test.c)
target_link_libraries(job_lsf_exclude_hosts_test res)
add_test(NAME job_lsf_exclude_hosts_test
         COMMAND job_lsf_exclude_hosts_test job_program NULL LOCAL)

add_executable(job_program_output job_queue/tests/job_program_output.c)
target_link_libraries(job_program_output ecl)

add_executable(job_queue_test job_queue/tests/job_job_queue_test.c)
target_link_libraries(job_queue_test res)
add_test(NAME job_queue_test
         COMMAND job_queue_test
                 $<TARGET_FILE:job_program_output>)
set_tests_properties(job_queue_test PROPERTIES LABELS "SLOW_1")

add_executable(job_queue_stress_task job_queue/tests/job_queue_stress_task.c)
target_link_libraries(job_queue_stress_task ecl)

add_executable(job_queue_stress_test job_queue/tests/job_queue_stress_test.c)
target_link_libraries(job_queue_stress_test res)
add_test(NAME job_queue_stress_test
         COMMAND job_queue_stress_test
                 $<TARGET_FILE:job_queue_stress_task>
                 False)
set_tests_properties(job_queue_stress_test PROPERTIES LABELS "SLOW_1")

add_test(NAME job_queue_user_exit
         COMMAND job_queue_stress_test
                 $<TARGET_FILE:job_queue_stress_task>
                 True)

add_executable(job_queue_timeout_test job_queue/tests/job_queue_timeout_test.c)
target_link_libraries(job_queue_timeout_test res)
add_test(NAME job_queue_timeout_test
         COMMAND job_queue_timeout_test $<TARGET_FILE:job_queue_stress_task>)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/qsub_emulators/
     DESTINATION ${EXECUTABLE_OUTPUT_PATH})

add_executable(job_torque_submit_test job_queue/tests/job_torque_submit_test.c)
target_link_libraries(job_torque_submit_test res)
add_test(NAME job_torque_submit_test
         WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
         COMMAND job_torque_submit_test dummyparam)
set_property(TEST job_torque_submit_test PROPERTY ENVIRONMENT “setenv PATH ${EXECUTABLE_OUTPUT_PATH}:$PATH”)

add_executable(ext_joblist_test job_queue/tests/ext_joblist_test.c)
target_link_libraries(ext_joblist_test res)
add_test(NAME ext_joblist_test
         COMMAND ext_joblist_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/tests/data/jobs/util
                 ${CMAKE_CURRENT_SOURCE_DIR})

# The testing of the lsf submit capabilities is quite troublesome for
# two reasons, and therefor by default disabled:
#
#
#  1. The shell based LSF commands require that user running the
#     bsub/bjobs/bxxx command has passwordless ssh configured to log in
#     to the lsf server. When the build and testing is run as a common
#     'jenkins' user this becomes difficult.
#
#  2. Submitting through the lsf library requires that the build/test
#     server actually is a LIM host; which it typically is not.
#
#-----------------------------------------------------------------
#
# This should be a space separated list of servers which will be
# tried out when testing the LSF submit capability. The test program
# will interpret the special strings 'NULL' and 'LOCAL' as follows:
#
#   NULL:  Submit using the linked in library functions.
#   LOCAL: Submit using shell commands on the current server
#
if (ERT_LSF_SUBMIT_TEST)
  set(LSF_SERVER "" CACHE STRING  "List of LSF servers for testing")

  if (LSF_LIBRARY)
     add_executable(job_lsb_test job_queue/tests/job_lsb_test.c)
     target_link_libraries(job_lsb_test res)
     add_test(NAME job_lsb_test COMMAND job_lsb_test)

     add_executable(job_lsb job_queue/tests/job_lsb.c)
     target_link_libraries(job_lsb res)
     add_test(NAME job_lsb COMMAND job_lsb)
   endif()

   add_executable(job_lsf_remote_submit_test job_queue/tests/job_lsf_remote_submit_test.c)
   target_link_libraries(job_lsf_remote_submit_test res)

   add_executable(job_lsf_library_submit_test job_queue/tests/job_lsf_library_submit_test.c)
   target_link_libraries(job_lsf_library_submit_test res)

   add_executable(job_program job_queue/tests/job_program.c)

   if (LSF_SERVER)
     add_test(NAME job_lsf_remote_submit_test
              COMMAND job_lsf_remote_submit_test
                      job_program
                      ${LSF_SERVER}
                      NULL
                      LOCAL)
   else()
      add_test(NAME job_lsf_remote_submit_test
               COMMAND job_lsf_remote_submit_test
                       job_program
                       NULL
                       LOCAL)
   endif()

   install(TARGETS job_program job_lsf_library_submit_test
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
#-----------------------------------------------------------------

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
                enkf_obs_invalid_path
                enkf_obs_tests
                enkf_obs_vector
                enkf_pca_plot
                enkf_plot_data
                enkf_plot_gendata
                enkf_plot_gen_kw
                enkf_plot_gen_kw_vector
                enkf_plot_genvector
                enkf_plot_tvector
                enkf_executable_path
                enkf_run_arg
                enkf_state_map
                obs_vector_tests
                log_config_level_parse
                value_export
                rng_manager
                rng_config
                trans_func
    )
    add_executable(${test} enkf/tests/${test}.c)
    target_link_libraries(${test} res)
    add_test(NAME ${test} COMMAND ${test})
endforeach ()

function( add_config_test name command )
    add_test( NAME ${name}
              COMMAND ${command} ${ARGN})

    set_property( TEST ${name} PROPERTY ENVIRONMENT "ERT_ROOT=${ERT_ROOT}")
endfunction()


foreach(test enkf_magic_string_in_workflows
             enkf_queue_config
             enkf_gen_obs_load
             enkf_ert_workflow_list
             enkf_main
             enkf_select_case_job
             enkf_plot_gen_kw_fs
             enkf_workflow_job_test_version
             enkf_ert_test_context
             gen_kw_logarithmic_test
             enkf_analysis_config_analysis_load
             enkf_forward_init_GEN_KW
             enkf_forward_init_GEN_PARAM
             enkf_umask_config_test
             enkf_forward_load_context
             enkf_ensemble_GEN_PARAM
             enkf_workflow_job_test2
             gen_kw_test
             enkf_runpath_list)

    add_executable(${test} enkf/tests/${test}.c)
    target_link_libraries(${test} res)
endforeach()

add_config_test(enkf_workflow_job_test2
                enkf_workflow_job_test2
                ${CMAKE_SOURCE_DIR}/test-data/local/snake_oil_no_data/snake_oil_GEN_DATA.ert
                ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/LOAD_RESULTS)

add_config_test(enkf_forward_load_context
                enkf_forward_load_context
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config
                forward_load_context)

add_config_test(enkf_ensemble_GEN_PARAM
                enkf_ensemble_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/ensemble/GEN_PARAM)


add_config_test(enkf_umask_config_test
                enkf_umask_config_test
                ${CMAKE_SOURCE_DIR}/test-data/local/simple_config/config_umask)

add_config_test(enkf_forward_init_GEN_PARAM_TRUE
                enkf_forward_init_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/forward/ert
                config_GEN_PARAM_true
                TRUE)

add_config_test(enkf_forward_init_GEN_PARAM_FALSE
                enkf_forward_init_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/forward/ert
                config_GEN_PARAM_false
                FALSE)

add_config_test(enkf_select_case_job
                enkf_select_case_job
                ${CMAKE_SOURCE_DIR}/test-data/local/snake_oil/snake_oil.ert
                ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal-tui/config/SELECT_CASE)
add_config_test(enkf_forward_init_GEN_KW_TRUE
                enkf_forward_init_GEN_KW
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/forward/ert
                config_GEN_KW_true
                TRUE)
add_config_test(enkf_forward_init_GEN_KW_FALSE
                enkf_forward_init_GEN_KW
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/forward/ert
                config_GEN_KW_false
                FALSE)
add_config_test(enkf_queue_config enkf_queue_config)
add_config_test(enkf_magic_string_in_workflows enkf_magic_string_in_workflows ${CMAKE_SOURCE_DIR}/test-data/local/config/workflows/config)
add_config_test(enkf_main enkf_main ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config rng)
add_config_test(enkf_runpath_list enkf_runpath_list ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/runpath_list/config)
add_config_test(enkf_gen_obs_load enkf_gen_obs_load ${CMAKE_SOURCE_DIR}/test-data/local/config/gen_data/config)
add_config_test(enkf_ert_workflow_list enkf_ert_workflow_list ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/SCALE_STD)
add_config_test(enkf_ert_test_context
                enkf_ert_test_context
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/test_context/config
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/test_context/wf_job
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/test_context/wf_job_fail)
add_config_test(enkf_plot_gen_kw_fs
                enkf_plot_gen_kw_fs
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/gen_kw_plot/config)
add_config_test(gen_kw_test
                gen_kw_test
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/forward/ert/config_GEN_KW_true)
add_config_test(gen_kw_logarithmic_test
                gen_kw_logarithmic_test
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/gen_kw_logarithmic/config_GEN_KW_logarithmic)





add_config_test(enkf_analysis_config_analysis_load enkf_analysis_config_analysis_load ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/analysis_load_config)
# It seems necessary to set the LD_LIBRARY_PATH for the loading of the
# rml_enkf module to work.  Do understand why!
set_property(TEST enkf_analysis_config_analysis_load PROPERTY
    ENVIRONMENT
    "ERT_SITE_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/config/analysis_load_site_config;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")

add_config_test(enkf_workflow_job_test_version enkf_workflow_job_test_version ${CMAKE_CURRENT_SOURCE_DIR}/enkf/tests/data/workflow_jobs)
#-----------------------------------------------------------------

if (NOT STATOIL_TESTDATA_ROOT)
    return ()
endif ()

add_test(NAME rms_file_test
         COMMAND rms_file_test ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/rms_file_test/rms_file)

add_test(NAME sched_history_summary1
         COMMAND sched_history_summary ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE)

add_test(NAME sched_history_summary2
         COMMAND sched_history_summary ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Snorre/SNORRE)


foreach(test enkf_site_config
             enkf_main_fs_current_file_test
             enkf_scale_correlated_std
             enkf_plot_gendata_fs
             enkf_state_report_step_compatible
             enkf_state_manual_load_test
             enkf_export_field_test
             enkf_workflow_job_test
             enkf_forward_init_SURFACE
             enkf_forward_init_FIELD
             enkf_forward_init_transform
             enkf_state_skip_summary_load_test
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
             enkf_main_fs )

    add_executable(${test} enkf/tests/${test}.c)
    target_link_libraries(${test} res)
endforeach()


function( add_statoil_test name command )
    add_test( NAME ${name}
              COMMAND ${command} ${ARGN})

    set_property( TEST ${name} PROPERTY LABELS StatoilData)
    set_property( TEST ${name} PROPERTY ENVIRONMENT "ERT_ROOT=${ERT_ROOT}")
endfunction()


add_statoil_test(enkf_site_config
                 enkf_site_config /project/res/etc/ERT/site-config)

add_statoil_test(enkf_gen_data_config
                 enkf_gen_data_config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/gendata_test/RFT_E-3H_21
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/gendata_test/RFT_E-3H_21_empty)

add_statoil_test(enkf_block_obs
                 enkf_block_obs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.EGRID)

add_statoil_test(enkf_obs_fs
                 enkf_obs_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/obs_testing/config)

add_statoil_test(enkf_obs_vector_fs
                 enkf_obs_vector_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/obs_testing/config)

add_statoil_test(enkf_plot_data_fs
                 enkf_plot_data_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/plotData/config)

add_statoil_test(enkf_time_map1 enkf_time_map)
add_statoil_test(enkf_time_map2
                 enkf_time_map
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/ModifiedSummary/EXTRA_TSTEP
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/ModifiedSummary/SHORT
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/ModifiedSummary/MISSING_TSTEP)

add_statoil_test(enkf_main_fs
                 enkf_main_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/plotData/config)

add_statoil_test(enkf_main_fs_current_file_test
                 enkf_main_fs_current_file_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/plotData/config)

add_statoil_test(enkf_scale_correlated_std
                 enkf_scale_correlated_std
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/with_data/config
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/STD_SCALE_CORRELATED_OBS)

add_statoil_test(enkf_plot_gendata_fs
                 enkf_plot_gendata_fs
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/with_GEN_DATA/config)

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

add_statoil_test(enkf_state_manual_load_test
                 enkf_state_manual_load_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_sum_compatible_true
                 config_ecl_sum_compatible_true)


add_statoil_test(enkf_state_summary_vars_present
                 enkf_state_skip_summary_load_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_summary_vars_config
                 config_summary_vars)

add_statoil_test(enkf_state_no_summary_vars_present
                 enkf_state_skip_summary_load_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_no_summary_vars_config
                 config_no_summary_vars)

add_statoil_test(enkf_export_field_test
                 enkf_export_field_test
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/export_fields/config
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_FIELD
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_FIELD_ECL_GRDECL
                 ${CMAKE_SOURCE_DIR}/share/workflows/jobs/internal/config/EXPORT_FIELD_RMS_ROFF)

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

add_statoil_test(enkf_export_inactive_cells
                 enkf_export_inactive_cells
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/export_inactive_cells/config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/export_inactive_cells/petro.grdecl)

add_statoil_test(enkf_refcase_list
                 enkf_refcase_list
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat*/ECLIPSE)

add_statoil_test(enkf_refcase_list2
                 enkf_refcase_list
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat*/ECLIPSE.*)

add_statoil_test(enkf_ecl_config1 enkf_ecl_config)
add_statoil_test(enkf_ecl_config2
                 enkf_ecl_config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE)

add_statoil_test(enkf_ecl_config_config
                 enkf_ecl_config_config
                 ${CMAKE_SOURCE_DIR}/test-data/Statoil/config/ecl_config)

