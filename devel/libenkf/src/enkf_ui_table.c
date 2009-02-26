#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <arg_pack.h>
#include <enkf_main.h>
#include <enkf_ui_plot.h>
#include <enkf_ui_fs.h>
#include <enkf_obs.h>
#include <field_obs.h>
#include <field_config.h>
#include <obs_vector.h>
#include <plot.h>
#include <plot_dataset.h>
#include <enkf_ui_util.h>
#include <ensemble_config.h>
#include <msg.h>




	
	   

static void enkf_ui_table__(enkf_main_type * enkf_main , bool ens_plot) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  const int ens_size = ensemble_config_get_size(ensemble_config);
  const int last_report = enkf_main_get_total_length( enkf_main );
  int iens1, iens2, step1 , step2;
  int num_keys , ikey;
  int length;
  int * index;
  char * input_keys;
  char ** user_keys;
  char ** index_keys;
  double ** data;
  enkf_config_node_type ** config_nodes;
  enkf_node_type        ** nodes;

  const int prompt_len = 50;
  const char * prompt  = "Table headings: KEY1:INDEX1   KEY2:INDEX2 ....";

  util_printf_prompt(prompt , prompt_len , '=' , "=> ");
  input_keys = util_alloc_stdin_line();
  util_split_string(input_keys , " " , &num_keys , &user_keys);

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
    step1  = util_scanf_int_with_limits("Report step",prompt_len , 0 , last_report);
    step2  = step1 + 1;
    length = (iens2 - iens1);
  } else {
    iens1  = util_scanf_int_with_limits("Ensemble member",prompt_len , 0 , ens_size - 1);
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
    state_enum state  = forecast;
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

    printf("\n\n"); 
    if (ens_plot)
      enkf_util_fprintf_data( index , (const double **) data , "Member #"   , (const char **) user_keys , active_length , num_keys , true , stdout);
    else
      enkf_util_fprintf_data( index , (const double **) data , "Report step" , (const char **) user_keys , active_length , num_keys , false , stdout);
  }

  for (ikey = 0; ikey < num_keys; ikey++) {
    enkf_node_free( nodes[ikey] );
    free(index_keys[ikey]);
    free(user_keys[ikey]);
    free(data[ikey]);
  }
  free( input_keys );
  free( user_keys );
  free( index_keys);
  free( data );
  free( nodes );
  free( config_nodes );
}





static void enkf_ui_table_ensemble(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_ui_table__(enkf_main , true);
}


static void enkf_ui_table_time(void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_ui_table__(enkf_main , false);
}


static void enkf_ui_table_misfit( void * arg) {
  const state_enum load_state = both;
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size = ensemble_config_get_size(ensemble_config);
  double * chi2 = util_malloc( ens_size * sizeof * chi2 , __func__);
  int iens;

  enkf_obs_total_ensemble_chi2( enkf_obs , fs , ens_size , load_state , chi2);
  //enkf_obs_ensemble_chi2( enkf_obs , fs , 30 , ens_size , load_state , chi2);

  printf("\n");
  printf(" ----------------------------------\n");
  printf(" Realization  |  Total chi^2 misfit\n");
  printf(" ----------------------------------\n");
  for (iens = 0; iens < ens_size; iens++)
    printf(" %11d  |       %g \n",iens , chi2[iens]);
  printf(" ----------------------------------\n\n");
  free( chi2 );
}






void enkf_ui_table_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  {
    menu_type * menu = menu_alloc("EnKF table menu" , "Back" , "bB");
    menu_add_item(menu , "Ensemble of parameters"          , "eE"  , enkf_ui_table_ensemble , enkf_main , NULL);
    menu_add_item(menu , "Time development of parameters"  , "tT"  , enkf_ui_table_time     , enkf_main , NULL);
    menu_add_item(menu , "Misfit table"                    , "mM"  , enkf_ui_table_misfit   , enkf_main , NULL);
    menu_add_separator(menu);
    menu_add_item(menu , "Manage cases" , "cC" , enkf_ui_fs_menu , enkf_main , NULL);
    menu_run(menu);
    menu_free(menu);
  }
}
