#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <arg_pack.h>
#include <enkf_main.h>
#include <enkf_tui_plot.h>
#include <enkf_tui_fs.h>
#include <enkf_obs.h>
#include <field_obs.h>
#include <field_config.h>
#include <obs_vector.h>
#include <plot.h>
#include <plot_dataset.h>
#include <enkf_tui_util.h>
#include <ensemble_config.h>
#include <msg.h>
#include <gen_kw_config.h>



	
	   

static void enkf_tui_table__(enkf_main_type * enkf_main , bool gen_kw_table , bool ens_plot) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const int ens_size    = ensemble_config_get_size(ensemble_config);
  const int last_report = enkf_main_get_total_length( enkf_main );
  int iens1, iens2, step1 , step2;
  int ikey , num_keys;
  int length;
  FILE * stream = NULL;
  int  * index;
  char ** user_keys;
  char ** index_keys;
  double ** data;
  enkf_config_node_type ** config_nodes;
  enkf_node_type        ** nodes;

  const int prompt_len = 50;
  const char * keylist_prompt  = "Table headings: KEY1:INDEX1   KEY2:INDEX2 ....";
  const char * gen_kw_prompt   = "GEN_KW Parameter";
  const char * file_prompt     = "File to save in (blank for nothing) ";
  
  if (gen_kw_table) {
    char * key;
    enkf_config_node_type * config_node;
    util_printf_prompt(gen_kw_prompt , prompt_len , '=' , "=> ");
    key  = util_alloc_stdin_line();
    if (ensemble_config_has_key( ensemble_config , key )) {
      config_node = ensemble_config_get_node( ensemble_config , key );
      if (enkf_config_node_get_impl_type( config_node ) == GEN_KW) {
        gen_kw_config_type *  gen_kw_config = enkf_config_node_get_ref( config_node );
        num_keys                            = gen_kw_config_get_data_size( gen_kw_config );
        
        user_keys = util_malloc( num_keys * sizeof * user_keys , __func__);
        for (int i=0; i < num_keys; i++) 
          user_keys[i] = gen_kw_config_alloc_user_key( gen_kw_config , i);
        
      } else {
        fprintf(stderr,"wrong type of: %s \n",key);
        free( key );
        return; /* going home on invalid input */
      }
    } else {
      fprintf(stderr,"** warning: do not have key:%s \n", key);
      free( key );
      return ; /* going home on invalid input */
    }
    free( key );
  } else {
    char * input_keys;
    util_printf_prompt(keylist_prompt , prompt_len , '=' , "=> ");
    input_keys = util_alloc_stdin_line();
    util_split_string(input_keys , " " , &num_keys , &user_keys);
    free( input_keys );
  }
  

  util_printf_prompt(file_prompt , prompt_len , '=' , "=> ");
  {
    char * filename = util_alloc_stdin_line( );
    if (filename != NULL)
      stream = util_mkdir_fopen( filename , "w");
    free( filename );
  }

  nodes        = util_malloc( num_keys * sizeof * nodes        , __func__);
  config_nodes = util_malloc( num_keys * sizeof * config_nodes , __func__);
  index_keys   = util_malloc( num_keys * sizeof * index_keys   , __func__);
  for (ikey  = 0; ikey < num_keys; ikey++) {
    config_nodes[ikey] = (enkf_config_node_type *) ensemble_config_user_get_node( ensemble_config , user_keys[ikey] , &index_keys[ikey]);
    nodes[ikey]        = enkf_node_alloc( config_nodes[ikey] );
  }
  
  if (ens_plot) {
    iens1  = 0;
    iens2  = ensemble_config_get_size(ensemble_config);
    step1  = util_scanf_int_with_limits("report step",prompt_len , 0 , last_report);
    step2  = step1 + 1;
    length = (iens2 - iens1);
  } else {
    iens1  = util_scanf_int_with_limits("ensemble member",prompt_len , 0 , ens_size - 1);
    iens2  = iens1 + 1;
    step1  = 0;
    step2  = last_report + 1;
    length = (step2 - step1);
  }
  index = util_malloc( length   * sizeof * index , __func__);
  data  = util_malloc( num_keys * sizeof * data  , __func__);
  {
    int i;
    for (i = 0; i < num_keys; i++)
      data[i] = util_malloc( length * sizeof * data[i] , __func__);
  }
  
  {
    state_enum state  = FORECAST;
    int active_length = 0;
    int total_line_count = 0;
    double line[num_keys];
    int iens, step;
    
    for (iens = iens1; iens < iens2; iens ++) {
      for (step = step1; step < step2; step++) {
	int line_count = 0;
	bool valid;
	
	for (ikey = 0; ikey < num_keys; ikey++) {
	  if (enkf_fs_has_node(fs , config_nodes[ikey] , step , iens , state)) {
	    enkf_fs_fread_node(fs , nodes[ikey] , step , iens , state);
	    line[ikey] = enkf_node_user_get( nodes[ikey] , index_keys[ikey] , &valid);
	    if (valid) 
	      line_count++;
	    else
	      line[ikey] = -1;
	  } 
	}
	
	if (line_count > 0) {
	  for (ikey=0; ikey < num_keys; ikey++) 
	    data[ikey][active_length] = line[ikey];
	  index[active_length] = total_line_count;
	  active_length++;
	}
	
	total_line_count++;
      }
    }
    
    if (stream != NULL) {
      if (ens_plot) 
        enkf_util_fprintf_data( index , (const double **) data , "Realization"   , (const char **) user_keys , active_length , num_keys , true , stream);
      else
        enkf_util_fprintf_data( index , (const double **) data , "Report-step" , (const char **) user_keys , active_length , num_keys , false , stream);
      fclose(stream);
    }

    printf("\n\n"); 
    if (ens_plot) 
      enkf_util_fprintf_data( index , (const double **) data , "Realization"   , (const char **) user_keys , active_length , num_keys , true , stdout);
    else
      enkf_util_fprintf_data( index , (const double **) data , "Report-step" , (const char **) user_keys , active_length , num_keys , false , stdout);
  }

  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_free( nodes[ikey] );
    free(index_keys[ikey]);
    free(user_keys[ikey]);
    free(data[ikey]);
  }
  free( user_keys );
  free( index_keys);
  free( data );
  free( nodes );
  free( config_nodes );
}





static void enkf_tui_table_ensemble(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_tui_table__(enkf_main , false , true);
}


static void enkf_tui_table_GEN_KW_ensemble(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_tui_table__(enkf_main , true , true);
}


static void enkf_tui_table_time(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_tui_table__(enkf_main , false , false);
}





void enkf_tui_table_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  {
    menu_type * menu = menu_alloc("Table of results" , "Back" , "bB");
    menu_add_item(menu , "Ensemble of parameters"          , "eE"  , enkf_tui_table_ensemble        , enkf_main , NULL);
    menu_add_item(menu , "GEN_KW ensemble"                 , "gG"  , enkf_tui_table_GEN_KW_ensemble , enkf_main , NULL);
    menu_add_item(menu , "Time development of parameters"  , "tT"  , enkf_tui_table_time            , enkf_main , NULL);
    menu_run(menu);
    menu_free(menu);
  }
}
