#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_tui_run.h>
#include <enkf_tui_export.h>
#include <enkf_tui_plot.h>
#include <enkf_tui_table.h>
#include <enkf_tui_fs.h>
#include <enkf_tui_ranking.h>
#include <enkf_main.h>
#include <enkf_sched.h>


/**
   This file implements the (text based) user interface in the enkf
   system.
*/



/** 
    The main loop.
*/





void enkf_tui_main_menu(enkf_main_type * enkf_main) {
  menu_type * menu = menu_alloc("Main menu" , "Quit" , "qQ");
  menu_add_item(menu , "Manage cases" , "cC" , enkf_tui_fs_menu , enkf_main , NULL);
  menu_add_item(menu , "Run or restart experiment"    , "rR" , enkf_tui_run_menu    , enkf_main , NULL);
  menu_add_item(menu , "Plot results"                 , "pP" , enkf_tui_plot_menu   , enkf_main , NULL);
  menu_add_item(menu , "Rank results"                 , "aA" , enkf_tui_ranking_menu, enkf_main , NULL);
  menu_add_item(menu , "Export data to other formats" , "eE" , enkf_tui_export_menu , enkf_main , NULL);
  menu_add_item(menu , "Table of results"             , "tT" , enkf_tui_table_menu  , enkf_main , NULL);
  menu_run(menu);
  menu_free(menu);
}


