#include <stdlib.h>
#include <string.h>
#include <menu.h>
#include <enkf_tui_util.h>
#include <enkf_tui_init.h>
#include <enkf_main.h>
#include <enkf_types.h>
#include <enkf_fs.h>
#include <arg_pack.h>
#include <util.h>
#include <msg.h>


void enkf_tui_fs_ls_case(void * arg) {
  enkf_fs_type    * fs      = (enkf_fs_type * ) arg;
  stringlist_type * dirlist = enkf_fs_alloc_dirlist( fs );
  int idir;

  printf("Avaliable cases: ");
  for (idir = 0; idir < stringlist_get_size( dirlist ); idir++)
    printf("%s ",stringlist_iget( dirlist , idir ));
  
  printf("\n");
  stringlist_free( dirlist );
}


void enkf_tui_fs_create_case(void * arg)
{
  int prompt_len = 50;
  char dir[256];
  char * menu_title;

  arg_pack_type  * arg_pack = arg_pack_safe_cast( arg );
  enkf_fs_type   * enkf_fs  = arg_pack_iget_ptr(arg_pack, 0);
  menu_type      * menu     = arg_pack_iget_ptr(arg_pack, 1);

  util_printf_prompt("Name of new case" , prompt_len , '=' , "=> ");
  scanf("%s", dir);
  
  if(enkf_fs_has_dir(enkf_fs, dir))
    printf("** WARNING **: Case \"%s\" already exists, will only select it.\n", dir);

  enkf_fs_select_write_dir(enkf_fs, dir, true);
  enkf_fs_select_read_dir( enkf_fs, dir      );

  menu_title = util_alloc_sprintf("Manage cases. Current: %s", enkf_fs_get_read_dir(enkf_fs));
  menu_set_title(menu, menu_title);
  free(menu_title);
}


/**
   Return NULL if no action should be performed.
*/
static char * enkf_tui_fs_alloc_existing_case(enkf_fs_type * fs , const char * prompt , int prompt_len) {
  char * name;
  while (true) {
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    name = util_alloc_stdin_line();

    if (name == NULL)  /* The user entered a blank string */
      break;
    else {
      if (enkf_fs_has_dir(fs , name)) 
	break;
      else {
	printf("** Can not find case: \"%s\" \n",name);
	free(name);
      }
    } 

  }
  
  return name;
}




void enkf_tui_fs_select_case(void * arg)
{
  int    prompt_len = 40;
  char * new_case;
  char * menu_title;

  arg_pack_type  * arg_pack = arg_pack_safe_cast( arg );
  enkf_fs_type   * enkf_fs  = arg_pack_iget_ptr(arg_pack, 0);
  menu_type      * menu     = arg_pack_iget_ptr(arg_pack, 1);
  new_case = enkf_tui_fs_alloc_existing_case( enkf_fs , "Name of case" , prompt_len);
  if (new_case != NULL) {
    enkf_fs_select_write_dir(enkf_fs, new_case, false);
    enkf_fs_select_read_dir( enkf_fs, new_case      );
    
    menu_title = util_alloc_sprintf("Manage cases. Current: %s", enkf_fs_get_read_dir(enkf_fs));
    menu_set_title(menu, menu_title);
    free(menu_title);
    free(new_case);
  }
}






static void enkf_tui_fs_copy_ensemble__(
  enkf_main_type * enkf_main,
  const char     * source_case,
  const char     * target_case,
  int              report_step_from,
  state_enum       state_from,
  int              report_step_to,
  state_enum       state_to,
  bool             only_parameters)
{
  enkf_fs_type   * fs        	= enkf_main_get_fs(enkf_main);
  msg_type       * msg       	= msg_alloc("Copying: ");
  ensemble_config_type * config = enkf_main_get_ensemble_config(enkf_main);
  int ens_size                  = enkf_main_get_ensemble_size(enkf_main);
  char * ranking_key;
  int  * ranking_permutation = NULL;
  int  * identity_permutation;

  misfit_table_type * misfit_table = enkf_main_get_misfit( enkf_main );
  
  if (misfit_table != NULL) {
    util_printf_prompt("Name of ranking to resort by (or blank)" , 50  , '=' , "=> ");
    ranking_key = util_alloc_stdin_line();
    if (misfit_table_has_ranking( misfit_table , ranking_key )) 
      ranking_permutation = misfit_table_get_ranking_permutation( misfit_table , ranking_key );
    else {
      fprintf(stderr," Sorry: ranking:%s does not exist \n", ranking_key );
      return;
    }
  }
  identity_permutation = util_malloc( ens_size * sizeof * identity_permutation , __func__ );
  {
    int iens;
    for (iens =0; iens < ens_size; iens++) 
      identity_permutation[iens] = iens;
  }

  if (ranking_permutation == NULL)
    ranking_permutation = identity_permutation;
  

  {
    /* Store current selections */
    char * user_read_dir  = util_alloc_string_copy(enkf_fs_get_read_dir( fs));
    char * user_write_dir = util_alloc_string_copy(enkf_fs_get_write_dir(fs));
    
    /* If the current target_case does not exist it is automatically created by the select_write_dir function */
    if( !enkf_fs_has_dir(fs, target_case)) 
      printf("Creating new case: %s \n",target_case);  
    
    enkf_fs_select_write_dir(fs, target_case, true );
    enkf_fs_select_read_dir( fs, source_case       );

    
    stringlist_type * nodes;
    if(only_parameters)
      nodes = ensemble_config_alloc_keylist_from_var_type(config, PARAMETER);
    else {
      /* Must explicitly load the static nodes. */
      
      stringlist_type * restart_kw_list = stringlist_alloc_new();
      int i;
      enkf_fs_fread_restart_kw_list(fs , report_step_from , 0 , restart_kw_list);  
      for (i = 0; i < stringlist_get_size( restart_kw_list ); i++) {
	const char * kw = stringlist_iget( restart_kw_list , i);
	if (!ensemble_config_has_key(config , kw)) 
	  ensemble_config_add_node(config , kw , STATIC_STATE , STATIC , NULL , NULL , NULL );
      }
      for (i=0; i < ens_size; i++) 
	enkf_fs_fwrite_restart_kw_list(fs , report_step_to , i , restart_kw_list);
      
      stringlist_free( restart_kw_list );
      nodes = ensemble_config_alloc_keylist(config);
    }

    /***/
    
    int num_nodes = stringlist_get_size(nodes);
    
    msg_show(msg);
    for(int i = 0; i < num_nodes; i++) {
      const char * key = stringlist_iget(nodes, i);
      enkf_config_node_type * config_node = ensemble_config_get_node(config , key);
      msg_update(msg , key);
      enkf_fs_copy_node_ensemble(fs, config_node, report_step_from, state_from, report_step_to , state_to , ens_size , ranking_permutation);
    }
    
    msg_free(msg , true);
    stringlist_free(nodes);
    
    /* Recover initial selections. */
    enkf_fs_select_write_dir(fs, user_write_dir, false);
    enkf_fs_select_read_dir( fs, user_read_dir        );
    free(user_read_dir);
    free(user_write_dir);
  }
  free( identity_permutation );
}





void enkf_tui_fs_initialize_case_from_copy(void * arg) 
{
  int prompt_len =50;
  char * source_case;
  int ens_size;
  int last_report;
  int src_step;
  state_enum src_state, target_state;
  enkf_main_type   * enkf_main = enkf_main_safe_cast( arg );
  enkf_fs_type     * fs        = enkf_main_get_fs(enkf_main);

  const ensemble_config_type * config = enkf_main_get_ensemble_config(enkf_main);
  ens_size = enkf_main_get_ensemble_size( enkf_main );

  
  last_report  = enkf_main_get_history_length( enkf_main );

  source_case = enkf_tui_fs_alloc_existing_case( fs , "Initialize from case" , prompt_len);
  if (source_case != NULL) {                                              
    char * ranking_key  = NULL;
    bool_vector_type * iens_mask = bool_vector_alloc( 0 , true ); 
    src_step         = util_scanf_int_with_limits("Source report step",prompt_len , 0 , last_report);
    src_state        = enkf_tui_util_scanf_state("Source analyzed/forecast [A|F]" , prompt_len , false);
    enkf_main_initialize_from_existing( enkf_main , source_case , src_step , src_state , iens_mask , ranking_key );
    bool_vector_free( iens_mask );
  }
  util_safe_free( source_case );
}



void enkf_tui_fs_copy_ensemble(void * arg)
{
  int prompt_len = 35;
  char * source_case;
  char * target_case;
  int last_report;
  int report_step_from;
  int report_step_to;
  state_enum state_from;
  state_enum state_to;

  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  enkf_fs_type   * fs        = enkf_main_get_fs(enkf_main);

  source_case = util_alloc_string_copy(enkf_fs_get_read_dir(fs));
  last_report  = enkf_main_get_history_length( enkf_main );

  report_step_from = util_scanf_int_with_limits("Source report step",prompt_len , 0 , last_report);
  state_from       = enkf_tui_util_scanf_state("Source analyzed/forecast [A|F]" , prompt_len , false);

  util_printf_prompt("Target case" , prompt_len , '=' , "=> ");
  target_case = util_alloc_stdin_line();


  report_step_to = util_scanf_int_with_limits("Target report step",prompt_len , 0 , last_report);
  state_to       = enkf_tui_util_scanf_state("Target analyzed/forecast [A|F]" , prompt_len , false);

  enkf_tui_fs_copy_ensemble__(enkf_main, source_case, target_case, report_step_from, state_from, report_step_to, state_to, false);

  free(target_case);
  free(source_case);
}



void enkf_tui_fs_copy_ensemble_of_parameters(void * arg)
{
  int prompt_len = 35;
  char * source_case;
  char * target_case;
  int last_report;
  int report_step_from;
  int report_step_to;
  state_enum state_from;
  state_enum state_to;

  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  enkf_fs_type   * fs        = enkf_main_get_fs(enkf_main);

  source_case  	   = util_alloc_string_copy(enkf_fs_get_read_dir(fs));
  last_report  	   = enkf_main_get_history_length( enkf_main );

  report_step_from = util_scanf_int_with_limits("Source report step",prompt_len , 0 , last_report);
  state_from       = enkf_tui_util_scanf_state("Source analyzed/forecast [A|F]" , prompt_len , false);
  
  util_printf_prompt("Target case" , prompt_len , '=' , "=> ");
  target_case = util_alloc_stdin_line();
  
  report_step_to = util_scanf_int_with_limits("Target report step",prompt_len , 0 , last_report);
  state_to       = enkf_tui_util_scanf_state("Target analyzed/forecast [A|F]" , prompt_len , false);

  enkf_tui_fs_copy_ensemble__(enkf_main, source_case, target_case, report_step_from, state_from, report_step_to, state_to, true);

  free(target_case);
  free(source_case);
}



void enkf_tui_fs_initialize_case_for_predictions(void * arg)
{
  int prompt_len = 35;
  char source_case[256];
  int report_step_from;
  int report_step_to;
  state_enum state_from;
  state_enum state_to;

  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  enkf_fs_type   * fs        = enkf_main_get_fs(enkf_main);

  state_from       = ANALYZED;
  state_to         = ANALYZED;
  report_step_from = enkf_main_get_history_length( enkf_main ); 
  report_step_to   = enkf_main_get_history_length( enkf_main );


  {
    char * target_case = util_alloc_string_copy(enkf_fs_get_write_dir(fs));
    
    util_printf_prompt("Source case" , prompt_len , '=' , "=> ");
    scanf("%s", source_case);
    
    enkf_tui_fs_copy_ensemble__(enkf_main, source_case, target_case, report_step_from, state_from, report_step_to, state_to, false);
    
    free(target_case);
  }
}



void enkf_tui_fs_menu(void * arg) {
  
   enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
   enkf_fs_type    * fs         = enkf_main_get_fs( enkf_main );

   const char * menu_title = util_alloc_sprintf("Manage cases - current: %s", enkf_fs_get_read_dir(fs));
   menu_type * menu = menu_alloc(menu_title , "Back" , "bB");

   menu_add_item(menu , "List available cases" , "lL" , enkf_tui_fs_ls_case , fs , NULL);
   
   {
     arg_pack_type * arg_pack = arg_pack_alloc();
     arg_pack_append_ptr(arg_pack  , fs);
     arg_pack_append_ptr(arg_pack  , menu);
     menu_add_item(menu , "Create and select new case" , "cC" , enkf_tui_fs_create_case, arg_pack , arg_pack_free__);
   }

   {
     arg_pack_type * arg_pack = arg_pack_alloc();
     arg_pack_append_ptr(arg_pack  , fs);
     arg_pack_append_ptr(arg_pack  , menu);
     menu_add_item(menu , "Select case" , "sS" , enkf_tui_fs_select_case, arg_pack , arg_pack_free__);
   }

   menu_add_separator(menu);
   menu_add_item(menu, "Initialize case from scratch"      		   , "iI" , enkf_tui_init_menu                          , enkf_main , NULL); 
   menu_add_item(menu, "Initialize case from existing case"		   , "aA" , enkf_tui_fs_initialize_case_from_copy       , enkf_main , NULL); 
   menu_add_item(menu, "Initialize case FOR PREDICTIONS from existing case", "pP" , enkf_tui_fs_initialize_case_for_predictions , enkf_main , NULL); 

   menu_add_separator(menu);
   /* Are these two in use??? */
   menu_add_item(menu, "Copy full ensemble to another case", "eE", enkf_tui_fs_copy_ensemble, enkf_main, NULL); 
   menu_add_item(menu, "Copy ensemble of parameters to another case", "oO", enkf_tui_fs_copy_ensemble_of_parameters, enkf_main, NULL); 

   menu_run(menu);
   menu_free(menu);
}

