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






void enkf_tui_run_start__(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
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
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  const int prompt_len  = 35;
  const int last_report = enkf_main_get_history_length( enkf_main );
  int start_report;
  state_enum state;
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }

  start_report = util_scanf_int_with_limits("Report step",prompt_len , 0 , last_report);
  state        = enkf_tui_util_scanf_state("Analyzed/forecast" , prompt_len , false);
  
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
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  int prompt_len = 45;
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);

  state_enum init_state    = ANALYZED; 
  int start_report   	   = 0;
  int init_step_parameters = 0;
  {
    char * prompt = util_alloc_sprintf("Which realizations to simulate[ensemble size:%d] : " , ens_size);
    char * select_string;
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    select_string = util_alloc_stdin_line();
    util_sscanf_active_range( select_string , ens_size - 1 , iactive);
    free( prompt );
    free( select_string );
  }
  
  enkf_main_run(enkf_main , ENSEMBLE_EXPERIMENT , iactive , init_step_parameters , start_report , init_state);
  free(iactive);
}




void enkf_main_interactive_set_runpath__(void *arg) {
  
}


void enkf_tui_run_analyze__(void * enkf_main) {
  int report_step = enkf_tui_util_scanf_report_step(enkf_main_get_history_length(enkf_main) , "Which report step to analyze" , 40);
  enkf_main_UPDATE(enkf_main , report_step - 1, report_step );
}


void enkf_tui_run_smooth__(void * enkf_main) {
  int last_report = enkf_main_get_history_length( enkf_main ) ;
  int step1 = enkf_tui_util_scanf_report_step(last_report , "First report step" , 20);
  int step2 = enkf_tui_util_scanf_report_step(last_report , "Last report step" , 20);

  if(step1 <= step2)
    enkf_main_UPDATE(enkf_main , step1, step2 );
}



void enkf_tui_run_predictions__(void * __enkf_main) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(__enkf_main);
  if (enkf_main_has_prediction( enkf_main )) {
    const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
    const int ens_size     			 = ensemble_config_get_size(ensemble_config);
    bool * iactive         			 = util_malloc(ens_size * sizeof * iactive , __func__);
    int        history_end 			 = enkf_main_get_history_length( enkf_main );
    state_enum start_state 			 = ANALYZED;           
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


void enkf_tui_run_manual_load__( void * arg ) {
  const int prompt_len                         = 60;
  enkf_main_type * enkf_main                   = enkf_main_safe_cast( arg );
  const int last_report                        = enkf_main_get_history_length( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size                           = ensemble_config_get_size(ensemble_config);
  int step1,step2;
  bool * iactive         = util_malloc(ens_size * sizeof * iactive , __func__);
  run_mode_type run_mode = ENSEMBLE_EXPERIMENT; //ENKF_ASSIMILATION; /*ENSEMBLE_EXPERIMENT;*/ /* Should really ask the user abourt this? */

  enkf_main_init_run(enkf_main , run_mode);     /* This is ugly */

  enkf_tui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
  {
    char * prompt = util_alloc_sprintf("Which realizations to load [ensemble size:%d] : " , ens_size);
    char * select_string;
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    select_string = util_alloc_stdin_line();
    util_sscanf_active_range( select_string , ens_size - 1 , iactive);
    free( prompt );
    free( select_string );
  }
  {
    int iens;
    arg_pack_type ** arg_list = util_malloc( ens_size * sizeof * arg_list , __func__);
    thread_pool_type * tp = thread_pool_alloc( 4 );  /* num_cpu - HARD coded. */

    for (iens = 0; iens < ens_size; iens++) {
      arg_pack_type * arg_pack = arg_pack_alloc();
      arg_list[iens] = arg_pack;
      
      if (iactive[iens]) {
        enkf_state_type * enkf_state = enkf_main_iget_state( enkf_main , iens );

        arg_pack_append_ptr( arg_pack , enkf_state);
        arg_pack_append_int( arg_pack , step1 );   /* This will be the load start parameter for the run_info struct. */
        arg_pack_append_int( arg_pack , step1 );
        arg_pack_append_int( arg_pack , step2 );
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
    char            * title      = util_alloc_sprintf("Run menu [case:%s]" , enkf_fs_get_read_dir(  enkf_main_get_fs( enkf_main ))) ;
    menu = menu_alloc(title , "Back" , "bB");
    free(title);
  }
  menu_add_item(menu , "Run ensemble experiment"                , "xX" , enkf_tui_run_exp__         , enkf_main , NULL);
  menu_add_separator( menu );
  menu_add_item(menu , "Start EnKF run from beginning"          , "sS" , enkf_tui_run_start__       , enkf_main , NULL);
  menu_add_item(menu , "Restart EnKF run from arbitrary state"  , "rR" , enkf_tui_run_restart__     , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Start predictions from end of history"  , "pP" , enkf_tui_run_predictions__ , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Load results manually"                  , "lL"  , enkf_tui_run_manual_load__ , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Analyze one step manually" , "aA" , enkf_tui_run_analyze__ , enkf_main , NULL);
  menu_add_item(menu , "Analyze interval manually" , "iI" , enkf_tui_run_smooth__  , enkf_main , NULL);
  //menu_add_separator(menu);
  //menu_add_item(menu , "Manually load simulation results" , "mM" , enkf_tui_run_manual_internalize__ , enkf_main , NULL);
  menu_add_separator(menu);
  {
    model_config_type * model_config = enkf_main_get_model_config( enkf_main );
    path_fmt_type     * runpath_fmt  = model_config_get_runpath_fmt( model_config );
    arg_pack_type * arg_pack = arg_pack_alloc();  
    char * runpath_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , path_fmt_get_fmt ( runpath_fmt ));
    
    arg_pack_append_ptr(arg_pack , model_config);
    arg_pack_append_ptr(arg_pack , menu_add_item(menu , runpath_label , "dD" , model_config_interactive_set_runpath__ , arg_pack , arg_pack_free__));
    
    
    free(runpath_label);
  }
  menu_run(menu);
  menu_free(menu);

}
