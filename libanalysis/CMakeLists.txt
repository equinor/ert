project(res-analysis C)

add_library(analysis src/analysis_module.c
                     src/bootstrap_enkf.c
                     src/cv_enkf.c
                     src/enkf_linalg.c
                     src/fwd_step_enkf.c
                     src/fwd_step_log.c
                     src/module_data_block.c
                     src/module_data_block_vector.c
                     src/module_info.c
                     src/module_obs_block.c
                     src/module_obs_block_vector.c
                     src/null_enkf.c
                     src/sqrt_enkf.c
                     src/std_enkf.c
)

target_include_directories(analysis
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)

target_link_libraries(analysis PUBLIC ecl)
target_compile_definitions(analysis PRIVATE -DINTERNAL_LINK)
set_target_properties(analysis PROPERTIES
    VERSION ${RES_VERSION_MAJOR}.${RES_VERSION_MINOR}
    SOVERSION ${RES_VERSION_MAJOR}
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS analysis
        EXPORT  res-config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_library(rml_enkf SHARED modules/rml_enkf_config.c
                            modules/rml_enkf.c
                            modules/rml_enkf_common.c
                            modules/rml_enkf_log.c
)
target_link_libraries(rml_enkf PRIVATE analysis ${CMAKE_DL_LIBS})
target_include_directories(rml_enkf PRIVATE modules)
set_target_properties(rml_enkf PROPERTIES VERSION 1.0
                                          SOVERSION 1.0
                                          PREFIX "")

add_library(std_enkf_debug SHARED modules/std_enkf_debug.c)
target_link_libraries(std_enkf_debug PRIVATE analysis ${CMAKE_DL_LIBS})
target_include_directories(std_enkf_debug PRIVATE modules)
set_target_properties(std_enkf_debug PROPERTIES VERSION 1.0
                                                SOVERSION 1.0
                                                PREFIX "")

install(TARGETS rml_enkf std_enkf_debug
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(PROGRAMS ert_module DESTINATION ${CMAKE_INSTALL_BINDIR})

if (NOT BUILD_TESTS)
    return ()
endif()

add_executable(analysis_external_module tests/analysis_test_external_module.c)
target_link_libraries(analysis_external_module analysis)

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

add_executable(analysis_test_module_info tests/analysis_test_module_info.c)
target_link_libraries(analysis_test_module_info analysis)
add_test(NAME analysis_test_module_info COMMAND analysis_test_module_info)
