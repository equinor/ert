#include <stdlib.h>
#include <menu.h>
#include <enkf_ui_util.h>
#include <enkf_main.h>
#include <enkf_types.h>
#include <enkf_fs.h>
#include <arg_pack.h>
#include <util.h>


void enkf_ui_fs_ls_case(void * arg) {
  enkf_fs_type    * fs      = (enkf_fs_type * ) arg;
  stringlist_type * dirlist = enkf_fs_alloc_dirlist( fs );
  int idir;

  printf("Avaliable cases: ");
  for (idir = 0; idir < stringlist_get_size( dirlist ); idir++)
    printf("%s ",stringlist_iget( dirlist , idir ));
  
  printf("\n");
  stringlist_free( dirlist );
}


void enkf_ui_fs_create_case(void * arg)
{
  char dir[256];
  char * menu_title;

  arg_pack_type  * arg_pack = arg_pack_safe_cast( arg );
  enkf_fs_type   * enkf_fs  = arg_pack_iget_ptr(arg_pack, 0);
  menu_type      * menu     = arg_pack_iget_ptr(arg_pack, 1);

  printf("Name of case ==> ");
  scanf("%s", dir);
  
  if(enkf_fs_has_dir(enkf_fs, dir))
    printf("** WARNING **: Case \"%s\" already exists, will only select it.\n", dir);

  enkf_fs_select_write_dir(enkf_fs, dir, true);
  enkf_fs_select_read_dir( enkf_fs, dir      );

  menu_title = util_alloc_sprintf("Manage cases. Current: %s", enkf_fs_get_read_dir(enkf_fs));
  menu_set_title(menu, menu_title);
  free(menu_title);
}



void enkf_ui_fs_select_case(void * arg)
{
  char dir[256];
  char * menu_title;

  arg_pack_type  * arg_pack = arg_pack_safe_cast( arg );
  enkf_fs_type   * enkf_fs  = arg_pack_iget_ptr(arg_pack, 0);
  menu_type      * menu     = arg_pack_iget_ptr(arg_pack, 1);

  printf("Name of case ==> ");
  scanf("%s", dir);
  
  if(!enkf_fs_has_dir(enkf_fs, dir))
  {
    printf("** ERROR **: Case \"%s\" does not exist.\n", dir);
  }
  else
  {
    enkf_fs_select_write_dir(enkf_fs, dir, false);
    enkf_fs_select_read_dir( enkf_fs, dir      );

    menu_title = util_alloc_sprintf("Manage cases. Current: %s", enkf_fs_get_read_dir(enkf_fs));
    menu_set_title(menu, menu_title);
    free(menu_title);
  }
}



void enkf_ui_fs_copy_all_parameters(void * arg)
{
  int prompt_len = 35;
  char * current_case;
  char target_case[256];
  int ens_size;
  int last_report;
  int report_step_from;
  int report_step_to;
  state_enum state_from;
  state_enum state_to;

  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  enkf_fs_type   * fs        = enkf_main_get_fs(enkf_main);

  const ensemble_config_type * config = enkf_main_get_ensemble_config(enkf_main);
  ens_size = ensemble_config_get_size(config);

  current_case = util_alloc_string_copy(enkf_fs_get_read_dir(fs));
  last_report  = enkf_main_get_total_length( enkf_main );

  /**
    Read user input and set read/write cases.
  */
  report_step_from = util_scanf_int_with_limits("Source report step",prompt_len , 0 , last_report);
  state_from       = enkf_ui_util_scanf_state("Source analyzed/forecast [A|F]" , prompt_len , false);


  /* 
     Simplify with some defaults. 
  */
  report_step_to = 0;
  if (report_step_to == 0)
    state_to = analyzed;
  else
    state_to = forecast;

  printf("Target case ==> ");
  scanf("%s", target_case);
  enkf_fs_select_write_dir( fs, target_case, true );

  {
    /**
      Copy that shit.
    */
    stringlist_type * parameters = ensemble_config_alloc_var_typed_keylist(config, parameter);
    int num_parameters = stringlist_get_size(parameters);

    for(int i = 0; i < num_parameters; i++)
    {
      const char * key = stringlist_iget(parameters, i);
      enkf_config_node_type * config_node = ensemble_config_get_node(config, key);
      enkf_fs_copy_ensemble(fs, config_node, report_step_from, state_from, report_step_to , state_to , 0, ens_size - 1);
    }

    stringlist_free(parameters);
  }


  /**
    Revert to original case.
  */
  enkf_fs_select_write_dir(fs, current_case, false);
  free(current_case);
}



void enkf_ui_fs_copy_ensemble(void * arg)
{
  int prompt_len = 35;
  char * current_case;
  char target_case[256];
  int ens_size;
  int last_report;
  int report_step_from;
  int report_step_to;
  state_enum state_from;
  state_enum state_to;

  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  enkf_fs_type   * fs        = enkf_main_get_fs(enkf_main);

  const ensemble_config_type * config = enkf_main_get_ensemble_config(enkf_main);
  ens_size = ensemble_config_get_size(config);

  current_case = util_alloc_string_copy(enkf_fs_get_read_dir(fs));
  last_report  = enkf_main_get_total_length( enkf_main );

  /**
    Read user input and set read/write cases.
  */
  report_step_from = util_scanf_int_with_limits("Source report step",prompt_len , 0 , last_report);
  state_from       = enkf_ui_util_scanf_state("Source analyzed/forecast [A|F]" , prompt_len , false);

  printf("Target case ==> ");
  scanf("%s", target_case);
  enkf_fs_select_write_dir( fs, target_case, true );

  report_step_to = util_scanf_int_with_limits("Target report step",prompt_len , 0 , last_report);
  state_to       = enkf_ui_util_scanf_state("Target analyzed/forecast [A|F]" , prompt_len , false);


  {
    /**
      Copy that shit.
    */
    int num_nodes;
    char ** nodes = ensemble_config_alloc_keylist(config, &num_nodes);

    for(int i = 0; i < num_nodes; i++)
    {
      const char * key = nodes[i];
      enkf_config_node_type * config_node = ensemble_config_get_node(config, key);
      enkf_fs_copy_ensemble(fs, config_node, report_step_from, state_from, report_step_to , state_to , 0, ens_size - 1);
    }

    util_free_stringlist(nodes, num_nodes);
  }


  /**
    Revert to original case.
  */
  enkf_fs_select_write_dir(fs, current_case, false);
  free(current_case);
}


void enkf_ui_fs_menu(void * arg) {
  
   enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
   enkf_fs_type    * fs         = enkf_main_get_fs( enkf_main );

   const char * menu_title = util_alloc_sprintf("Manage cases. Current: %s", enkf_fs_get_read_dir(fs));
   menu_type * menu = menu_alloc(menu_title , "Back" , "bB");

   menu_add_item(menu , "List available cases" , "lL" , enkf_ui_fs_ls_case , fs , NULL);
   
   {
     arg_pack_type * arg_pack = arg_pack_alloc();
     arg_pack_append_ptr(arg_pack  , fs);
     arg_pack_append_ptr(arg_pack  , menu);
     menu_add_item(menu , "Create case" , "cC" , enkf_ui_fs_create_case, arg_pack , arg_pack_free__);
   }

   {
     arg_pack_type * arg_pack = arg_pack_alloc();
     arg_pack_append_ptr(arg_pack  , fs);
     arg_pack_append_ptr(arg_pack  , menu);
     menu_add_item(menu , "Select case" , "sS" , enkf_ui_fs_select_case, arg_pack , arg_pack_free__);
   }

   menu_add_item(menu, "Copy parameters to new case", "pP", enkf_ui_fs_copy_all_parameters, enkf_main, NULL); 
   menu_add_item(menu, "Copy ensemble", "eE", enkf_ui_fs_copy_ensemble, enkf_main, NULL); 

   menu_run(menu);
   menu_free(menu);
}

