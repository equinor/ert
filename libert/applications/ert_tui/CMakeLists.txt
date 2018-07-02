include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

set( src_list main.cpp enkf_tui_main.cpp  enkf_tui_fs.cpp  enkf_tui_ranking.cpp  enkf_tui_misc.cpp  enkf_tui_table.cpp menu.cpp
              enkf_tui_run.cpp enkf_tui_util.cpp  enkf_tui_init.cpp  enkf_tui_export.cpp  enkf_tui_analysis.cpp enkf_tui_help.cpp enkf_tui_simple.cpp enkf_tui_workflow.cpp)

set_source_files_properties( main.cpp PROPERTIES COMPILE_DEFINITIONS "COMPILE_TIME_STAMP=\"${ERT_BUILD_TIME}\";GIT_COMMIT=\"${GIT_COMMIT}\"")

add_executable( ert_tui ${src_list} )
target_link_libraries( ert_tui res::res ert )
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
