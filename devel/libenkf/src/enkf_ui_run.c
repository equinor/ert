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

  enkf_main_run(enkf_main , iactive , 0 , analyzed);
  free(iactive);
}



void enkf_ui_run_restart__(void * enkf_main) {
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  const int prompt_len  = 35;
  const int last_report = enkf_sched_get_last_report(enkf_sched);
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
  
  enkf_main_run(enkf_main ,  iactive , start_report , state);
  free(iactive);
}


void enkf_ui_run_exp__(void * enkf_main) {
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const bool load_results = false;
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size           = ensemble_config_get_size(ensemble_config);
  const int last_report = enkf_sched_get_last_report(enkf_sched);
  int prompt_len = 45;
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);

  int start_report = util_scanf_int_with_limits("Initialize static parameters from: ",prompt_len , 0 , last_report );
  int small_ens_size = util_scanf_int_with_limits("How many members too integrate: ",prompt_len , 1 , ens_size);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++) {
      if (iens < small_ens_size)
	iactive[iens] = true;
      else
	iactive[iens] = false;
    }
  }
  enkf_main_run_step(enkf_main , ensemble_experiment , iactive , start_report , analyzed , 0 , last_report , load_results , false , enkf_sched_get_default_forward_model(enkf_sched));
  free(iactive);
}



void enkf_ui_run_screening__(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size  = ensemble_config_get_size(ensemble_config);
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const bool load_results = false;
  const int last_report   = enkf_sched_get_last_report(enkf_sched);
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  {
    int iens;
    for (iens= 0; iens < ens_size; iens++)
      iactive[iens] = true;
  }
    
  enkf_main_run_step(enkf_main , screening_experiment , iactive , 0 , analyzed , 0 , last_report , load_results , false , enkf_sched_get_default_forward_model(enkf_sched));
  free(iactive);
}


void enkf_main_interactive_set_runpath__(void *arg) {
  
}


void enkf_ui_run_menu(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );
  
  menu_type * menu = menu_alloc("EnKF run menu" , "qQ");
  menu_add_item(menu , "Start EnKF run from beginning"          , "sS" , enkf_ui_run_start__      , enkf_main);
  menu_add_item(menu , "Restart EnKF run from arbitrary state"  , "rR" , enkf_ui_run_restart__    , enkf_main);
  menu_add_item(menu , "Run ensemble experiment"                , "xX" , enkf_ui_run_exp__        , enkf_main);
  menu_add_item(menu , "Run screening experiment"               , "cC" , enkf_ui_run_screening__  , enkf_main);
  menu_add_separator(menu);
  {
    model_config_type * model_config = enkf_main_get_model_config( enkf_main );
    path_fmt_type     * runpath_fmt  = model_config_get_runpath_fmt( model_config );
    arg_pack_type * arg_pack = arg_pack_alloc();
    char * runpath_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , path_fmt_get_fmt ( runpath_fmt ));
    
    arg_pack_append_ptr(arg_pack , model_config);
    arg_pack_append_ptr(arg_pack , menu_add_item(menu , runpath_label , "pP" , model_config_interactive_set_runpath__ , arg_pack));
    
    
    free(runpath_label);
  }

  menu_run(menu);
  menu_free(menu);

}
