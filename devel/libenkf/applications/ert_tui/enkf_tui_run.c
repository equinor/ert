/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'enkf_tui_run.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <thread_pool.h>
#include <enkf_main.h>
#include <enkf_fs.h>
#include <enkf_sched.h>
#include <arg_pack.h>
#include <ensemble_config.h>
#include <enkf_analysis.h>
#include <enkf_tui_util.h>
#include <enkf_tui_fs.h>
#include <enkf_tui_analysis.h>
#include <ert_tui_const.h>



static void enkf_tui_run_set_runpath(void * arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  model_config_type * model_config = arg_pack_iget_ptr(arg_pack , 0);
  menu_item_type    * item         = arg_pack_iget_ptr(arg_pack , 1);
  char runpath_fmt[256];
  printf("Give runpath format ==> ");
  scanf("%s" , runpath_fmt);
  model_config_set_runpath_fmt(model_config , runpath_fmt);
  {
    char * menu_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , runpath_fmt);
    menu_item_set_label( item , menu_label );
    free(menu_label);
  }
}




void enkf_tui_run_start__(void * enkf_main) {
  const int ens_size = enkf_main_get_ensemble_size( enkf_main );
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }

  enkf_main_run(enkf_main , ENKF_ASSIMILATION , iactive , 0 , 0 , ANALYZED);
  free(iactive);
}



void enkf_tui_run_restart__(void * enkf_main) {
  const int ens_size           = enkf_main_get_ensemble_size( enkf_main );
  const int last_report = enkf_main_get_history_length( enkf_main );
  int start_report;
  state_enum state;
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }

  start_report = util_scanf_int_with_limits("Report step",PROMPT_LEN , 0 , last_report);
  state        = enkf_tui_util_scanf_state("Analyzed/forecast" , PROMPT_LEN , false);
  
  enkf_main_run(enkf_main , ENKF_ASSIMILATION , iactive , start_report , start_report  , state);
  free(iactive);
}


/** 
    Experiments will always start with the parameters at time == 0; if
    you want to simulate with updated (posterior) parameters, you
    ensure that by initializing from a report_step > 0 from an
    existing case.

    Prediction part is included if it exists.
*/



void enkf_tui_run_exp__(void * enkf_main) {
  const int ens_size           = enkf_main_get_ensemble_size( enkf_main );
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);

  state_enum init_state    = ANALYZED; 
  int start_report         = 0;
  int init_step_parameters = 0;
  {
    char * prompt = util_alloc_sprintf("Which realizations to simulate <default:all> : " , ens_size);
    char * select_string;
    util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
    select_string = util_alloc_stdin_line();
    if (select_string != NULL) {
      util_sscanf_active_range( select_string , ens_size - 1 , iactive);
      free( select_string );
    } else {
      /* The user entered <return> : Run all realizations. */
      int i;
      for (i=0; i < ens_size; i++)
        iactive[i] = true;
    }
    free( prompt );
  }
  
  enkf_main_run(enkf_main , ENSEMBLE_EXPERIMENT , iactive , init_step_parameters , start_report , init_state);
  free(iactive);
}



void enkf_tui_run_create_runpath__(void * __enkf_main) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(__enkf_main);
  const int ens_size           = enkf_main_get_ensemble_size( enkf_main );
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);

  state_enum init_state    = ANALYZED; 
  int start_report         = 0;
  int init_step_parameters = 0;
  {
    char * prompt = util_alloc_sprintf("Which realizations to create[ensemble size:%d] : " , ens_size);
    char * select_string;
    util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
    select_string = util_alloc_stdin_line();
    util_sscanf_active_range( select_string , ens_size - 1 , iactive);
    free( prompt );
    free( select_string );
  }

  enkf_main_run(enkf_main , ENSEMBLE_EXPERIMENT , iactive , init_step_parameters , start_report , init_state);
  free(iactive);
}






void enkf_tui_run_predictions__(void * __enkf_main) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(__enkf_main);
  if (enkf_main_has_prediction( enkf_main )) {
    const int ens_size                           = enkf_main_get_ensemble_size( enkf_main );
    bool * iactive                               = util_malloc(ens_size * sizeof * iactive , __func__);
    int        history_end                       = enkf_main_get_history_length( enkf_main );
    state_enum start_state                       = ANALYZED;           
    {
      int iens;
      for (iens= 0; iens < ens_size; iens++)
        iactive[iens] = true;
    }
    enkf_main_run(enkf_main , ENSEMBLE_PREDICTION , iactive , history_end , history_end  , start_state);
    free( iactive );

  } else
    fprintf(stderr,"** Sorry: you must set a schedule prediction file with configuration option SCHEDULE_PREDICTION_FILE to use this option.\n");
}


/**
   Runs an ensemble experiment including both history and prediction period. 
*/
void enkf_tui_run_full__(void * __enkf_main) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(__enkf_main);
  if (enkf_main_has_prediction( enkf_main )) {
    const int ens_size                           = enkf_main_get_ensemble_size( enkf_main );
    bool * iactive                               = util_malloc(ens_size * sizeof * iactive , __func__);
    state_enum init_state                        = ANALYZED; 
    int start_report                             = 0;
    int init_step_parameters                     = 0;                
    {
      int iens;
      for (iens= 0; iens < ens_size; iens++)
        iactive[iens] = true;
    }
    enkf_main_run(enkf_main , ENSEMBLE_PREDICTION , iactive , init_step_parameters , start_report , init_state);
    free( iactive );

  } else
    fprintf(stderr,"** Sorry: you must set a schedule prediction file with configuration option SCHEDULE_PREDICTION_FILE to use this option.\n");
}



void enkf_tui_run_manual_load__( void * arg ) {
  enkf_main_type * enkf_main                   = enkf_main_safe_cast( arg );
  enkf_fs_type * fs                            = enkf_main_get_fs( enkf_main ); 
  const int last_report                        = enkf_main_get_history_length( enkf_main );
  const int ens_size                           = enkf_main_get_ensemble_size( enkf_main );
  int step1,step2;
  bool * iactive         = util_malloc(ens_size * sizeof * iactive , __func__);
  run_mode_type run_mode = ENSEMBLE_EXPERIMENT; //ENSEMBLE_PREDICTION will induce the most powerfull load. ENKF_ASSIMILATION; /*ENSEMBLE_EXPERIMENT;*/ /* Should really ask the user abourt this? */

  enkf_main_init_run(enkf_main , run_mode);     /* This is ugly */
  /** Observe that for the summary data it will load all the available data anyway. */
  step1 = 0;
  step2 = last_report;
  {
    char * prompt = util_alloc_sprintf("Which realizations to load [ensemble size:%d] : " , ens_size);
    char * select_string;
    util_printf_prompt(prompt , PROMPT_LEN , '=' , "=> ");
    select_string = util_alloc_stdin_line();
    util_sscanf_active_range( select_string , ens_size - 1 , iactive);
    free( prompt );
    free( select_string );
  }
  {
    int iens;
    arg_pack_type ** arg_list = util_malloc( ens_size * sizeof * arg_list , __func__);
    thread_pool_type * tp = thread_pool_alloc( 4 , true );  /* num_cpu - HARD coded. */

    for (iens = 0; iens < ens_size; iens++) {
      arg_pack_type * arg_pack = arg_pack_alloc();
      arg_list[iens] = arg_pack;
      
      if (iactive[iens]) {
        enkf_state_type * enkf_state = enkf_main_iget_state( enkf_main , iens );

        arg_pack_append_ptr( arg_pack , enkf_state);
        arg_pack_append_ptr( arg_pack , fs );
        arg_pack_append_int( arg_pack , step1 );      /* This will be the load start parameter for the run_info struct. */
        arg_pack_append_int( arg_pack , step1 );      /* Step1 */ 
        arg_pack_append_int( arg_pack , step2 );      /* Step2 For summary data it will load the whole goddamn thing anyway.*/
        arg_pack_append_bool( arg_pack , true );      /* Verbose */
        thread_pool_add_job( tp , enkf_state_internalize_results_mt , arg_pack);
      }
    }
    
    thread_pool_join( tp );
    thread_pool_free( tp );
    
    for (iens = 0; iens < ens_size; iens++) 
      arg_pack_free( arg_list[iens]);
    free( arg_list );      
  }
  free( iactive );
}





void enkf_tui_run_menu(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );
  menu_type       * menu;
  {
    char            * title      = util_alloc_sprintf("Run menu [case:%s]" , enkf_main_get_current_fs( enkf_main ));
    menu = menu_alloc(title , "Back" , "bB");
    free(title);
  }
  menu_add_item(menu , "Ensemble run: history"                , "xX" , enkf_tui_run_exp__         , enkf_main , NULL);
  menu_add_item(menu , "Ensemble run: predictions"            , "pP" , enkf_tui_run_predictions__ , enkf_main , NULL);
  menu_add_item(menu , "Ensemble run: history + predictions"  , "fF" , enkf_tui_run_full__        , enkf_main , NULL);
  menu_add_separator( menu );
  menu_add_item(menu , "Start EnKF run from beginning"          , "sS" , enkf_tui_run_start__       , enkf_main , NULL);
  menu_add_item(menu , "Restart EnKF run from arbitrary state"  , "rR" , enkf_tui_run_restart__     , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Create runpath directories - NO simulation" , "cC" , enkf_tui_run_create_runpath__ , enkf_main , NULL );
  menu_add_item(menu , "Load results manually"                  , "lL"  , enkf_tui_run_manual_load__ , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Analysis menu"             , "aA" , enkf_tui_analysis_menu , enkf_main , NULL);
  menu_add_separator(menu);
  {
    model_config_type * model_config = enkf_main_get_model_config( enkf_main );
    path_fmt_type     * runpath_fmt  = model_config_get_runpath_fmt( model_config );
    arg_pack_type * arg_pack = arg_pack_alloc();  
    char * runpath_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , path_fmt_get_fmt ( runpath_fmt ));
    
    arg_pack_append_ptr(arg_pack , model_config);
    arg_pack_append_ptr(arg_pack , menu_add_item(menu , runpath_label , "dD" , enkf_tui_run_set_runpath , arg_pack , arg_pack_free__));
    
    free(runpath_label);
  }
  menu_run(menu);
  menu_free(menu);

}
