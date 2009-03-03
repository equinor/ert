#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <arg_pack.h>
#include <enkf_ui_util.h>
#include <enkf_ui_fs.h>
#include <ensemble_config.h>






void enkf_ui_run_start__(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }

  enkf_main_run(enkf_main , enkf_assimilation , iactive , -1 , 0 , analyzed);
  free(iactive);
}



void enkf_ui_run_restart__(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  const int prompt_len  = 35;
  const int last_report = enkf_main_get_total_length( enkf_main );
  int start_report;
  state_enum state;
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }

  start_report = util_scanf_int_with_limits("Report step",prompt_len , 0 , last_report);
  state        = enkf_ui_util_scanf_state("Analyzed/forecast" , prompt_len , false);
  
  enkf_main_run(enkf_main , enkf_assimilation , iactive , -1 , start_report  , state);
  free(iactive);
}


void enkf_ui_run_exp__(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  int prompt_len = 45;
  const int last_report   = enkf_main_get_total_length( enkf_main );
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);

  state_enum init_state = analyzed; 
  int start_report   	= 0;
  int init_report    	= util_scanf_int_with_limits("Initialize static parameters from: ",prompt_len , 0 , last_report );
  int iens1,iens2;
  enkf_ui_util_scanf_iens_range("Realizations to simulate (0 - %d): " , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++) {
      if (iens < iens)
	iactive[iens] = false;
      else if (iens > iens2)
	iactive[iens] = false;
      else
	iactive[iens] = true;
    }
  }

  enkf_main_run(enkf_main , ensemble_experiment , iactive , init_report , start_report , init_state);
  free(iactive);
}



void enkf_ui_run_screening__(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size      = ensemble_config_get_size(ensemble_config);
  bool * iactive          = util_malloc(ens_size * sizeof * iactive , __func__);
  int init_report  = 0;
  int start_report = 0;
  state_enum init_state = analyzed;
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }
  
  enkf_main_run(enkf_main , screening_experiment , iactive , init_report , start_report , init_state);
  free(iactive);
}


void enkf_main_interactive_set_runpath__(void *arg) {
  
}


void enkf_ui_run_analyze__(void * enkf_main) {
  int report_step = enkf_ui_util_scanf_report_step(enkf_main_get_total_length(enkf_main) , "Which report step to analyze" , 40);
  enkf_main_analysis_update(enkf_main , report_step - 1, report_step );
}


void enkf_ui_run_smooth__(void * enkf_main) {
  int last_report = enkf_main_get_total_length( enkf_main ) ;
  int step1 = enkf_ui_util_scanf_report_step(last_report , "First report step" , 20);
  int step2 = enkf_ui_util_scanf_report_step(last_report , "Last report step" , 20);

  if(step1 <= step2)
    enkf_main_analysis_update(enkf_main , step1, step2 );
}



void enkf_ui_run_predictions__(void * __enkf_main) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(__enkf_main);
  if (enkf_main_has_prediction( enkf_main )) {
    const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
    const int ens_size     			 = ensemble_config_get_size(ensemble_config);
    bool * iactive         			 = util_malloc(ens_size * sizeof * iactive , __func__);
    int        history_end 			 = enkf_main_get_history_length( enkf_main ) - 1;
    state_enum start_state 			 = analyzed;           
    {
      int iens;
      for (iens= 0; iens < ens_size; iens++)
	iactive[iens] = true;
    }
    enkf_main_run(enkf_main , ensemble_experiment , iactive , history_end , history_end  , start_state);
    free( iactive );

  } else
    fprintf(stderr,"** Sorry: you must set a schedule prediction file with configuration option SCHEDULE_PREDICTION_FILE to use this option.\n");
  

}






/**
   This implementation is NOT compatible with the general case where
   RUNPATH contains STEP1 and STEP2.
*/

//void enkf_ui_run_manual_internalize__(void * arg ) {
//  int load_mask1 = dynamic_state + dynamic_result + static_state;
//  int load_mask2 = dynamic_state + dynamic_result + static_state;
//  int step1 = 0;
//  int step2 = 30; /* inclusive */
//  int iens1 = 0;
//  int iens2 = 9; /* inclusive */
//  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
//  enkf_state_type ** ensemble = enkf_main_get_ensemble( enkf_main );
//  int iens;
//
//  for (iens = iens1; iens <= iens2; iens++) {
//    printf("Loading for member: %d \n",iens);
//    enkf_state_init_run( ensemble[iens] , enkf_assimilation , true , step1 , analyzed , step1 , step2 , NULL);
//    enkf_state_internalize_results( ensemble[iens] , load_mask1 , load_mask2 , step1 , step2);
//  }
//}



void enkf_ui_run_menu(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );
  
  menu_type * menu = menu_alloc("Run menu" , "Back" , "bB");
  menu_add_item(menu , "Run screening experiment"               , "eE" , enkf_ui_run_screening__   , enkf_main , NULL);
  menu_add_item(menu , "Run ensemble experiment"                , "xX" , enkf_ui_run_exp__         , enkf_main , NULL);
  menu_add_separator( menu );
  menu_add_item(menu , "Start EnKF run from beginning"          , "sS" , enkf_ui_run_start__       , enkf_main , NULL);
  menu_add_item(menu , "Restart EnKF run from arbitrary state"  , "rR" , enkf_ui_run_restart__     , enkf_main , NULL);
  menu_add_item(menu , "Start predictions from end of history"  , "pP" , enkf_ui_run_predictions__ , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Analyze one step manually" , "aA" , enkf_ui_run_analyze__ , enkf_main , NULL);
  menu_add_item(menu , "Analyze interval manually" , "iI" , enkf_ui_run_smooth__  , enkf_main , NULL);
  //menu_add_separator(menu);
  //menu_add_item(menu , "Manually load simulation results" , "mM" , enkf_ui_run_manual_internalize__ , enkf_main , NULL);
  menu_add_separator(menu);
  {
    model_config_type * model_config = enkf_main_get_model_config( enkf_main );
    path_fmt_type     * runpath_fmt  = model_config_get_runpath_fmt( model_config );
    arg_pack_type * arg_pack = arg_pack_alloc();  /* This will leak  */
    char * runpath_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , path_fmt_get_fmt ( runpath_fmt ));
    
    arg_pack_append_ptr(arg_pack , model_config);
    arg_pack_append_ptr(arg_pack , menu_add_item(menu , runpath_label , "dD" , model_config_interactive_set_runpath__ , arg_pack , arg_pack_free__));
    
    
    free(runpath_label);
  }
  menu_run(menu);
  menu_free(menu);

}
