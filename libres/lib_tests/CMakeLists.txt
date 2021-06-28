foreach(name ies_enkf_config ies_enkf_data)
  add_executable(${name} analysis/modules/${name}.cpp ${ies_source})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/lib/analysis/modules)
  target_link_libraries(${name} res ies)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

add_executable( ies_module analysis/modules/ies_enkf_module.cpp)
target_link_libraries(ies_module res ies)
add_test(NAME ies_module COMMAND ies_module $<TARGET_FILE_NAME:ies> ${CMAKE_CURRENT_SOURCE_DIR}/analysis/modules/test-data/poly)

add_executable(ies_linalg analysis/modules/ies_linalg.cpp ${ies_source})
target_include_directories(ies_linalg PRIVATE ${CMAKE_SOURCE_DIR}/lib/analysis/modules)
target_link_libraries(ies_linalg res ies)
add_test(NAME ies_linalg COMMAND ies_linalg ${CMAKE_CURRENT_SOURCE_DIR}/analysis/modules/test-data/poly)

add_executable(ies_std_compare analysis/modules/ies_std_compare.cpp ${ies_source})
target_include_directories(ies_std_compare PRIVATE ${CMAKE_SOURCE_DIR}/lib/analysis/modules)
target_link_libraries(ies_std_compare res ies)
add_test(NAME ies_std_compare COMMAND ies_std_compare ${CMAKE_CURRENT_SOURCE_DIR}/analysis/modules/test-data/poly)

add_executable(ies_iteration analysis/modules/ies_iteration.cpp ${ies_source})
target_include_directories(ies_iteration PRIVATE ${CMAKE_SOURCE_DIR}/lib/analysis/modules)
target_link_libraries(ies_iteration res ies)
add_test(NAME ies_iteration COMMAND ies_iteration ${CMAKE_CURRENT_SOURCE_DIR}/analysis/modules/test-data/poly_normal)

foreach(name ert_util_logh
             ert_util_arg_pack
             ert_util_matrix
             es_testdata
             ert_util_matrix_lapack
             ert_util_subst_list
             ert_util_subst_list_add_from_string
             ert_util_block_fs
             test_thread_pool
             res_util_PATH)

       add_executable(${name} res_util/${name}.cpp)
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

    add_executable(${name} config/${name}.cpp)
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

    add_executable(${name} config/${name}.cpp)
    target_link_libraries(${name} res)
endforeach()

macro(config_test name input_file)
   add_test(NAME ${name} COMMAND ${name} ${CMAKE_CURRENT_SOURCE_DIR}/config/data/${input_file})
endmacro()

config_test( config_append_test append_test)
config_test( config_node_test append_test)
config_test( config_typeOK type_testOK)
config_test( config_typeFail type_testFail)
config_test( config_content_item content_item_test)
config_test( config_define define_test)


add_test(NAME config_root_path
         COMMAND config_root_path ${CMAKE_CURRENT_SOURCE_DIR}/config/data)

add_test(NAME config_include_test
         COMMAND config_include_test ${CMAKE_CURRENT_SOURCE_DIR}/config/data include_test)

add_test(NAME config_argc
         COMMAND config_argc
            ${CMAKE_CURRENT_SOURCE_DIR}/config/data/argc_OK
            ${CMAKE_CURRENT_SOURCE_DIR}/config/data/argc_less
            ${CMAKE_CURRENT_SOURCE_DIR}/config/data/argc_more)



#-----------------------------------------------------------------

add_executable(sched_history_summary sched/sched_history_summary.cpp)
target_link_libraries(sched_history_summary res)

add_executable(rms_file_test rms/rms_file_test.cpp)
target_link_libraries(rms_file_test res)

add_executable(analysis_external_module analysis/analysis_test_external_module.cpp)
target_link_libraries(analysis_external_module res)

add_test(NAME analysis_module_rml
         COMMAND analysis_external_module
                 "RML_ENKF"
                 $<TARGET_FILE_NAME:rml_enkf> 41
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


foreach(name analysis_test_module_info analysis_module_test)
  add_executable(${name} analysis/${name}.cpp)
  target_link_libraries(${name} res)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

#-----------------------------------------------------------------


foreach(name job_status_test
             ext_job_test
             job_node_test
             job_lsf_parse_bsub_stdout
             job_lsf_test
             job_queue_driver_test
             job_slurm_driver
             job_mock_slurm
             job_torque_test)

    add_executable(${name} job_queue/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/lib/private-include)
    target_link_libraries(${name} res)
    add_test(NAME ${name} COMMAND ${name})
endforeach()

find_program(SBATCH "sbatch")
foreach(name job_slurm_submit
             job_slurm_runtest)
  add_executable(${name} job_queue/${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/lib/private-include)
  target_link_libraries(${name} res)
  if (SBATCH)
    add_test(NAME ${name} COMMAND ${name})
  endif()
endforeach()


foreach(name job_loadOK
             ext_joblist_test
             job_torque_submit_test
             job_queue_stress_task
             job_program_output
             job_lsf_exclude_hosts_test
             job_workflow_test
             create_file
             job_loadFail)

     add_executable(${name} job_queue/${name}.cpp)
     target_link_libraries(${name} res)
endforeach()

if (NOT APPLE)

     add_executable(job_list_test job_queue/job_list_test.cpp)
     target_link_libraries(job_list_test res)

     add_test(NAME job_list_test
         COMMAND valgrind --leak-check=full --error-exitcode=1
                 $<TARGET_FILE:job_list_test>)
endif()


add_test(NAME job_loadOK1
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/internalOK)
add_test(NAME job_loadOK2
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/externalOK)
add_test(NAME job_loadOK3
         COMMAND job_loadOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/internalOK
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/externalOK)

add_test(NAME job_loadFail1
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/internalFail)
add_test(NAME job_loadFail2
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/externalFail)
add_test(NAME job_loadFail3
         COMMAND job_loadFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/internalFail
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/externalFail)

set_target_properties(job_workflow_test PROPERTIES ENABLE_EXPORTS ON)
add_test(NAME job_workflow_test
         COMMAND job_workflow_test
                 ${CMAKE_CURRENT_BINARY_DIR}
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/internal_job)

add_test(NAME job_lsf_exclude_hosts_test
         COMMAND job_lsf_exclude_hosts_test job_program NULL LOCAL)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/qsub_emulators/
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME job_torque_submit_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND job_torque_submit_test dummyparam)
set_property(TEST job_torque_submit_test PROPERTY ENVIRONMENT “setenv PATH ${CMAKE_CURRENT_BINARY_DIR}:$PATH”)

add_test(NAME ext_joblist_test
         COMMAND ext_joblist_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/job_queue/data/jobs/util
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
     add_executable(job_lsb_test job_queue/job_lsb_test.cpp)
     target_link_libraries(job_lsb_test res)
     add_test(NAME job_lsb_test COMMAND job_lsb_test)

     add_executable(job_lsb job_queue/job_lsb.cpp)
     target_link_libraries(job_lsb res)
     add_test(NAME job_lsb COMMAND job_lsb)
   endif()

   add_executable(job_lsf_remote_submit_test job_queue/job_lsf_remote_submit_test.cpp)
   target_link_libraries(job_lsf_remote_submit_test res)

   add_executable(job_lsf_library_submit_test job_queue/job_lsf_library_submit_test.cpp)
   target_link_libraries(job_lsf_library_submit_test res)

   add_executable(job_program job_queue/job_program.cpp)

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
                enkf_ensemble
                enkf_ensemble_config
                enkf_ert_run_context
                enkf_fs
                enkf_gen_data_config_parse
                enkf_iter_config
                enkf_local_dataset
                enkf_local_obsdata
                enkf_local_obsdata_node
                enkf_meas_data
                enkf_model_config
                enkf_obs_invalid_path
                enkf_obs_tests
                enkf_obs_vector
                enkf_plot_data
                enkf_plot_gendata
                enkf_plot_gen_kw
                enkf_plot_gen_kw_vector
                enkf_plot_genvector
                enkf_plot_tvector
                enkf_update_log_test
                enkf_run_arg
                enkf_state_map
                obs_vector_tests
                log_config_level_parse
                row_scaling
                rng_manager
                rng_config
                trans_func
                value_export
    )
    add_executable(${test} enkf/${test}.cpp)
    target_link_libraries(${test} res)
    add_test(NAME ${test} COMMAND ${test})
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/lib/private-include)
endforeach ()

function( add_config_test name command )
    add_test( NAME ${name}
              COMMAND ${command} ${ARGN})

    set_property( TEST ${name} PROPERTY ENVIRONMENT "ERT_ROOT=${ERT_ROOT};ERT_SITE_CONFIG=${PROJECT_SOURCE_DIR}/${SHARE_DIR}/site-config")
endfunction()


foreach(test enkf_magic_string_in_workflows
             enkf_queue_config
             enkf_gen_obs_load
             enkf_ert_workflow_list
             enkf_main
             enkf_select_case_job
             enkf_plot_gen_kw_fs
             enkf_workflow_job_test_directory
             enkf_ert_test_context
             gen_kw_logarithmic_test
             enkf_analysis_config_analysis_load
             enkf_forward_init_GEN_KW
             enkf_forward_init_GEN_PARAM
             enkf_umask_config_test
             enkf_executable_path
             enkf_forward_load_context
             enkf_ensemble_GEN_PARAM
             enkf_workflow_job_test2
             gen_kw_test
             enkf_runpath_list)

    add_executable(${test} enkf/${test}.cpp)
    target_link_libraries(${test} res)
endforeach()

add_config_test(enkf_executable_path enkf_executable_path)

add_config_test(enkf_workflow_job_test2
                enkf_workflow_job_test2
                ${CMAKE_SOURCE_DIR}/../test-data/local/snake_oil_no_data/snake_oil_GEN_DATA.ert
                ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/LOAD_RESULTS)

add_config_test(enkf_forward_load_context
                enkf_forward_load_context
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config
                forward_load_context)

add_config_test(enkf_ensemble_GEN_PARAM
                enkf_ensemble_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/ensemble/GEN_PARAM)


add_config_test(enkf_umask_config_test
                enkf_umask_config_test
                ${CMAKE_SOURCE_DIR}/../test-data/local/simple_config/config_umask)

add_config_test(enkf_forward_init_GEN_PARAM_TRUE
                enkf_forward_init_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/forward/ert
                config_GEN_PARAM_true
                TRUE)

add_config_test(enkf_forward_init_GEN_PARAM_FALSE
                enkf_forward_init_GEN_PARAM
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/forward/ert
                config_GEN_PARAM_false
                FALSE)

add_config_test(enkf_select_case_job
                enkf_select_case_job
                ${CMAKE_SOURCE_DIR}/../test-data/local/snake_oil/snake_oil.ert
                ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal-tui/config/SELECT_CASE)
add_config_test(enkf_forward_init_GEN_KW_TRUE
                enkf_forward_init_GEN_KW
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/forward/ert
                config_GEN_KW_true
                TRUE)
add_config_test(enkf_forward_init_GEN_KW_FALSE
                enkf_forward_init_GEN_KW
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/forward/ert
                config_GEN_KW_false
                FALSE)
add_config_test(enkf_queue_config enkf_queue_config)
add_config_test(enkf_magic_string_in_workflows enkf_magic_string_in_workflows ${CMAKE_SOURCE_DIR}/../test-data/local/config/workflows/config)
add_config_test(enkf_main enkf_main ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config rng)
add_config_test(enkf_runpath_list enkf_runpath_list ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/runpath_list/config)
add_config_test(enkf_gen_obs_load enkf_gen_obs_load ${CMAKE_SOURCE_DIR}/../test-data/local/config/gen_data/config)
add_config_test(enkf_ert_workflow_list enkf_ert_workflow_list ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/SCALE_STD)
add_config_test(enkf_ert_test_context
                enkf_ert_test_context
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/test_context/config
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/test_context/wf_job
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/test_context/wf_job_fail)
add_config_test(enkf_plot_gen_kw_fs
                enkf_plot_gen_kw_fs
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/gen_kw_plot/config)
add_config_test(gen_kw_test
                gen_kw_test
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/forward/ert/config_GEN_KW_true)
add_config_test(gen_kw_logarithmic_test
                gen_kw_logarithmic_test
                ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/gen_kw_logarithmic/config_GEN_KW_logarithmic)

add_config_test(enkf_analysis_config_analysis_load enkf_analysis_config_analysis_load ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/analysis_load_config)
# It seems necessary to set the LD_LIBRARY_PATH for the loading of the
# rml_enkf module to work.  Do understand why!
if (UNIX AND NOT APPLE)
set_property(TEST enkf_analysis_config_analysis_load PROPERTY
    ENVIRONMENT
    "ERT_SITE_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/analysis_load_site_config;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")
elseif (APPLE)
set_property(TEST enkf_analysis_config_analysis_load PROPERTY
    ENVIRONMENT
    "ERT_SITE_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/config/analysis_load_site_config_osx;DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{DYLD_LIBRARY_PATH}")
endif()

add_config_test(enkf_workflow_job_test_directory enkf_workflow_job_test_directory ${CMAKE_CURRENT_SOURCE_DIR}/enkf/data/workflow_jobs)
#-----------------------------------------------------------------

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

    add_executable(${test} enkf/${test}.cpp)
    target_link_libraries(${test} res)
endforeach()


if (NOT EQUINOR_TESTDATA_ROOT)
    return ()
endif ()

add_test(NAME rms_file_test
         COMMAND rms_file_test ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/rms_file_test/rms_file)

add_test(NAME sched_history_summary1
         COMMAND sched_history_summary ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE)

add_test(NAME sched_history_summary2
         COMMAND sched_history_summary ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Snorre/SNORRE)



function( add_equinor_test name command )
    add_test( NAME ${name}
              COMMAND ${command} ${ARGN})

    set_property( TEST ${name} PROPERTY LABELS EquinorData)
    set_property( TEST ${name} PROPERTY ENVIRONMENT "ERT_ROOT=${ERT_ROOT}")
endfunction()


add_equinor_test(enkf_site_config
                 enkf_site_config /project/res/etc/ERT/site-config)

add_equinor_test(enkf_gen_data_config
                 enkf_gen_data_config
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/gendata_test/RFT_E-3H_21
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/gendata_test/RFT_E-3H_21_empty)

add_equinor_test(enkf_block_obs
                 enkf_block_obs
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE.EGRID)

add_equinor_test(enkf_obs_fs
                 enkf_obs_fs
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/obs_testing/config)

add_equinor_test(enkf_obs_vector_fs
                 enkf_obs_vector_fs
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/obs_testing/config)

add_equinor_test(enkf_plot_data_fs
                 enkf_plot_data_fs
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/plotData/config)

add_equinor_test(enkf_time_map1 enkf_time_map)
add_equinor_test(enkf_time_map2
                 enkf_time_map
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/ModifiedSummary/EXTRA_TSTEP
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/ModifiedSummary/SHORT
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/ModifiedSummary/MISSING_TSTEP)

add_equinor_test(enkf_main_fs
                 enkf_main_fs
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/plotData/config)

add_equinor_test(enkf_main_fs_current_file_test
                 enkf_main_fs_current_file_test
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/plotData/config)

add_equinor_test(enkf_scale_correlated_std
                 enkf_scale_correlated_std
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/with_data/config
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/STD_SCALE_CORRELATED_OBS)

add_equinor_test(enkf_plot_gendata_fs
                 enkf_plot_gendata_fs
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/with_GEN_DATA/config)

add_equinor_test(enkf_state_report_step_compatible_TRUE
                 enkf_state_report_step_compatible
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/ecl_sum_compatible_true
                 config_ecl_sum_compatible_true
                 TRUE)

add_equinor_test(enkf_state_report_step_compatible_FALSE
                 enkf_state_report_step_compatible
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/ecl_sum_compatible_false
                 config_ecl_sum_compatible_false
                 FALSE)

add_equinor_test(enkf_state_manual_load_test
                 enkf_state_manual_load_test
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/ecl_sum_compatible_true
                 config_ecl_sum_compatible_true)


add_equinor_test(enkf_state_summary_vars_present
                 enkf_state_skip_summary_load_test
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/ecl_summary_vars_config
                 config_summary_vars)

add_equinor_test(enkf_state_no_summary_vars_present
                 enkf_state_skip_summary_load_test
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/ecl_no_summary_vars_config
                 config_no_summary_vars)

add_equinor_test(enkf_export_field_test
                 enkf_export_field_test
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/export_fields/config
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_FIELD
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_FIELD_ECL_GRDECL
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_FIELD_RMS_ROFF)

add_equinor_test(enkf_workflow_job_test
                 enkf_workflow_job_test
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/with_data/config
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/enkf_state_runpath/config_runpath_multiple_iterations
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal-tui/config/CREATE_CASE
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal-tui/config/INIT_CASE_FROM_EXISTING
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/LOAD_RESULTS
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/LOAD_RESULTS_ITER
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/OBSERVATION_RANKING
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/DATA_RANKING
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_RANKING
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/INIT_MISFIT_TABLE
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/EXPORT_RUNPATH
                 ${CMAKE_SOURCE_DIR}/${SHARE_DIR}/workflows/jobs/internal/config/PRE_SIMULATION_COPY)


add_equinor_test(enkf_forward_init_SURFACE_TRUE
                 enkf_forward_init_SURFACE
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/surface config_surface_true
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/surface/Surface.irap
                 TRUE)

add_equinor_test(enkf_forward_init_SURFACE_FALSE
                 enkf_forward_init_SURFACE
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/surface config_surface_false
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/surface/Surface.irap
                 FALSE)

add_equinor_test(enkf_forward_init_FIELD_TRUE
                 enkf_forward_init_FIELD
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/field config_field_true
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/field/petro.grdecl
                 TRUE)

add_equinor_test(enkf_forward_init_FIELD_FALSE
                 enkf_forward_init_FIELD
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/field config_field_false
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/forward_init/field/petro.grdecl
                 FALSE)

add_equinor_test(enkf_forward_init_transform_TRUE
                 enkf_forward_init_transform
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/transform transform_forward_init_true
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/transform/petro.grdecl
                 TRUE)

add_equinor_test(enkf_forward_init_transform_FALSE
                 enkf_forward_init_transform
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/transform transform_forward_init_false
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/transform/petro.grdecl
                 FALSE)

add_equinor_test(enkf_export_inactive_cells
                 enkf_export_inactive_cells
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/export_inactive_cells/config
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/export_inactive_cells/petro.grdecl)

add_equinor_test(enkf_refcase_list
                 enkf_refcase_list
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat*/ECLIPSE)

add_equinor_test(enkf_refcase_list2
                 enkf_refcase_list
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat*/ECLIPSE.*)

add_equinor_test(enkf_ecl_config1 enkf_ecl_config)
add_equinor_test(enkf_ecl_config2
                 enkf_ecl_config
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/ECLIPSE/Gurbat/ECLIPSE)

add_equinor_test(enkf_ecl_config_config
                 enkf_ecl_config_config
                 ${CMAKE_SOURCE_DIR}/../test-data/Equinor/config/ecl_config)
