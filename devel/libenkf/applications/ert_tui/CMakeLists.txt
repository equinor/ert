include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
include_directories( ${libplot_src_path} )
include_directories( ${PLPLOT_INCLUDE_PATH} )

set( SITE_CONFIG_FILE "/project/res/etc/ERT/site-config" CACHE FILEPATH "Path to global ERT Configuration file")

set( src_list main.c enkf_tui_main.c  enkf_tui_fs.c  enkf_tui_ranking.c  enkf_tui_misc.c  enkf_tui_table.c  enkf_tui_plot.c enkf_tui_plot_rft.c enkf_tui_plot_util.c
              enkf_tui_run.c enkf_tui_util.c  enkf_tui_init.c  enkf_tui_export.c  enkf_tui_analysis.c enkf_tui_QC.c enkf_tui_help.c)



exec_program( svnversion ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE SVN_VERSION)
exec_program( date OUTPUT_VARIABLE COMPILE_TIME_STAMP)

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/CMake/config/ert_build_config_file.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/ert_build_info.h )

add_executable( ert ${src_list} )

target_link_libraries( ert enkf sched rms ecl geometry config plot job_queue analysis util )

#set(program_list convert.x kw_list.x grid_info.x summary.x)
#foreach(prog ${program_list})
#   target_link_libraries( ${prog} ecl_static util_static )
#   #-----------------------------------------------------------------
#   if (ECL_INSTALL_PREFIX)
#      set (destination ${INSTALL_ROOT}/${ECL_INSTALL_PREFIX}/bin)
#   else()
#      set (destination ${INSTALL_ROOT}/bin)
#   endif()
#
#   install(TARGETS ${prog} DESTINATION ${destination})
#   if (INSTALL_GROUP)
#      install(CODE "EXECUTE_PROCESS(COMMAND chgrp ${INSTALL_GROUP} ${destination}/${prog})")
#      install(CODE "EXECUTE_PROCESS(COMMAND chmod g+w ${destination}/${prog})")
#   endif()
#endforeach()
