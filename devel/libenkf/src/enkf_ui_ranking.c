#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <enkf_main.h>
#include <menu.h>
#include <enkf_obs.h>
#include <enkf_ui_util.h>



static void enkf_ui_ranking_make_misfit_table( void * arg) {
  const state_enum load_state  		       = both;
  enkf_main_type  * enkf_main  		       = enkf_main_safe_cast( arg );  
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int history_length                     = enkf_main_get_history_length( enkf_main );
  const int ens_size                           = ensemble_config_get_size(ensemble_config);
  misfit_table_type * misfit_table             = misfit_table_alloc( ensemble_config , fs , history_length , ens_size , enkf_obs );
  enkf_main_set_misfit_table( enkf_main , misfit_table );
}


static void enkf_ui_ranking_create( void * arg ) {
  enkf_main_type    * enkf_main       = enkf_main_safe_cast( arg );  
  enkf_obs_type     * enkf_obs        = enkf_main_get_obs( enkf_main );
  misfit_table_type * misfit_table    = enkf_main_get_misfit( enkf_main );
  
  if (misfit_table == NULL) {
    fprintf(stderr,"Sorry: must create a misfit table first \n");
    return;
  } else {
    const int history_length    = enkf_main_get_history_length( enkf_main );
    const int    prompt_len = 50;
    const char * prompt1  = "Observations to use for ranking";
    const char * prompt2  = "Name to store ranking under";
    int step1,step2;
    stringlist_type * ranking_keys;
    char * obs_keys_input;
    char * ranking_key;
    util_printf_prompt(prompt1 , prompt_len , '=' , "=> ");
    obs_keys_input = util_alloc_stdin_line();
    ranking_keys   = enkf_obs_alloc_matching_keylist( enkf_obs , obs_keys_input );
    enkf_ui_util_scanf_report_steps(history_length , prompt_len , &step1 , &step2);
    
    util_printf_prompt(prompt2 , prompt_len , '=' , "=> ");
    ranking_key = util_alloc_stdin_line();
    if (ranking_key == NULL) 
      ranking_key = util_alloc_string_copy( MISFIT_DEFAULT_RANKING_KEY);
    
    if (stringlist_get_size( ranking_keys ) > 0) {
      misfit_table_create_ranking( misfit_table  , ranking_keys , step1 , step2 , ranking_key);
      misfit_table_display_ranking( misfit_table , ranking_key);
    } else
      fprintf(stderr,"The input string : \"%s\" did not resolve to any valid observation keys \n", obs_keys_input);
    
    free( obs_keys_input );
    stringlist_free( ranking_keys );
    free( ranking_key );
  }
}


static void enkf_ui_ranking_display( void * arg ) {
  enkf_main_type    * enkf_main       = enkf_main_safe_cast( arg );  
  misfit_table_type * misfit_table    = enkf_main_get_misfit( enkf_main );

  if (misfit_table == NULL) {
    fprintf(stderr,"Sorry: must create a misfit table first \n");
    return;
  } else {
    const int prompt_len  = 50;
    const char * prompt1  = "Ranking to display";
    char * ranking_key;
    
    util_printf_prompt(prompt1 , prompt_len , '=' , "=> ");
    ranking_key    = util_alloc_stdin_line();
    if (ranking_key == NULL) 
      ranking_key = util_alloc_string_copy( MISFIT_DEFAULT_RANKING_KEY);
    if (misfit_table_has_ranking( misfit_table , ranking_key))
      misfit_table_display_ranking( misfit_table , ranking_key);
    else
      fprintf(stderr,"Sorry: could not find ranking key: %s \n", ranking_key );
    
    free( ranking_key );
  }
}


void enkf_ui_ranking_menu(void * arg) {
    
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  {
    menu_type * menu = menu_alloc("Ranking of results" , "Back" , "bB");
    menu_add_item(menu , "Create/update misfit table" , "cC"  , enkf_ui_ranking_make_misfit_table   , enkf_main , NULL);
    menu_add_separator( menu );
    menu_add_item(menu , "New ranking" , "nN" , enkf_ui_ranking_create  , enkf_main , NULL);
    menu_add_item(menu , "Display ranking"    , "dD" , enkf_ui_ranking_display , enkf_main , NULL);
    
    menu_run(menu);
    menu_free(menu);
  }
  
}
