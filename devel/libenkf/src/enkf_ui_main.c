#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_ui_run.h>
#include <enkf_ui_export.h>
#include <enkf_ui_plot.h>
#include <enkf_ui_init.h>
#include <enkf_ui_table.h>
#include <enkf_ui_fs.h>
#include <enkf_main.h>
#include <enkf_sched.h>


/**
   This file implements the (text based) user interface in the enkf
   system.
*/



/** 
    The main loop.
*/





void enkf_ui_main_menu(enkf_main_type * enkf_main) {
  menu_type * menu = menu_alloc("EnKF main menu" , "qQ");
  
  menu_add_item(menu , "Initialize EnKF ensemble"     , "iI" , enkf_ui_init_menu   , enkf_main , NULL);
  menu_add_item(menu , "Run EnKF"                     , "rR" , enkf_ui_run_menu    , enkf_main , NULL);
  menu_add_item(menu , "Export data to other formats" , "eE" , enkf_ui_export_menu , enkf_main , NULL);
  menu_add_item(menu , "Plot results"                 , "pP" , enkf_ui_plot_menu   , enkf_main , NULL);
  menu_add_item(menu , "Table of results"             , "tT" , enkf_ui_table_menu  , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Change directories for reading and writing" , "cC" , enkf_ui_fs_menu , enkf_main , NULL);
  menu_run(menu);
  menu_free(menu);
}


