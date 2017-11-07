include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

set( src_list main.c enkf_tui_main.c  enkf_tui_fs.c  enkf_tui_ranking.c  enkf_tui_misc.c  enkf_tui_table.c menu.c 
              enkf_tui_run.c enkf_tui_util.c  enkf_tui_init.c  enkf_tui_export.c  enkf_tui_analysis.c enkf_tui_help.c enkf_tui_simple.c enkf_tui_workflow.c)

set_source_files_properties( main.c PROPERTIES COMPILE_DEFINITIONS "COMPILE_TIME_STAMP=\"${ERT_BUILD_TIME}\";GIT_COMMIT=\"${GIT_COMMIT}\"")

add_executable( ert_tui ${src_list} )
target_link_libraries( ert_tui res::enkf ert )
if (USE_RUNPATH)
   add_runpath( ert_tui )
endif()

target_include_directories(ert_tui
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
)


set (destination ${CMAKE_INSTALL_PREFIX}/bin)

install(TARGETS ert_tui DESTINATION ${destination})
if (INSTALL_GROUP)
   install(CODE "EXECUTE_PROCESS(COMMAND chgrp ${INSTALL_GROUP} ${destination}/ert_tui)")
   install(CODE "EXECUTE_PROCESS(COMMAND chmod g+w ${destination}/ert_tui)")
endif()
