#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <enkf_types.h>
#include <enkf_ui_util.h>
#include <enkf_ui_init.h>
#include <enkf_state.h>
#include <enkf_node.h>
#include <enkf_fs.h>
#include <msg.h>
#include <ensemble_config.h>

static void enkf_ui_init__(enkf_main_type * enkf_main , const stringlist_type * param_list , int iens1 , int iens2) {
  int iens;
  enkf_fs_type * fs = enkf_main_get_fs(enkf_main);
  msg_type * msg = msg_alloc("Initializing...: " );
  msg_show(msg);

  for (iens = iens1; iens <= iens2; iens++) {
    int ip;
    enkf_state_type * state = enkf_main_iget_state( enkf_main , iens);
    {
      char * iens_string = util_alloc_sprintf("%04d" , iens);
      msg_update(msg , iens_string); 
      free(iens_string);
    }
    for (ip = 0; ip < stringlist_get_size(param_list); ip++) {
      enkf_node_type * param_node = enkf_state_get_node( state , stringlist_iget( param_list , ip));
      if (enkf_node_initialize( param_node , iens)) 
	enkf_fs_fwrite_node(fs , param_node , 0 , iens , analyzed);
    }
    msg_update(msg , "Done");
  }
  msg_free(msg , false);
}




void enkf_ui_init(enkf_main_type * enkf_main, bool all_members , bool all_parameters) {
  const int prompt_len                         = 35;
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  int   ens_size                               = ensemble_config_get_size(ensemble_config);
  stringlist_type * param_list                 = stringlist_alloc_new();
  int iens1, iens2;
  
  /* iens2 should be interpreted as __inclusive__ */
  if ( all_members ) {
    iens1 = 0;
    iens2 = ens_size - 1;
  } else {
    iens1 = util_scanf_int_with_limits("Initialize ensemble member" , prompt_len , 0 , ens_size - 1);
    iens2 = iens1;
  }

  if (all_parameters) {
    int num_keys;
    char ** tmp_key_list = ensemble_config_alloc_keylist( ensemble_config , &num_keys);
    for (int i = 0; i < num_keys; i++) {
      if (ensemble_config_var_type(ensemble_config , tmp_key_list[i]) != static_state)
	stringlist_append_copy(param_list , tmp_key_list[i]);
    }
    util_free_stringlist( tmp_key_list , num_keys );
  } else 
    stringlist_append_copy( param_list , enkf_config_node_get_key_ref(enkf_ui_util_scanf_parameter(ensemble_config , prompt_len , false , INVALID , invalid , NULL , NULL , NULL)) );

    
  enkf_ui_init__(enkf_main , param_list , iens1 , iens2);
  stringlist_free( param_list );
}


static void enkf_ui_init1(void * enkf_main) {
  enkf_ui_init(enkf_main, true , true);
}

static void enkf_ui_init2(void * enkf_main) {
  enkf_ui_init(enkf_main , true , false);
}

static void enkf_ui_init3(void * enkf_main) {
  enkf_ui_init(enkf_main , false , true );
}

static void enkf_ui_init4(void * enkf_main) {
  enkf_ui_init(enkf_main , false , false);
}


void enkf_ui_init_menu(void * arg) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(arg);

  menu_type * menu = menu_alloc("EnKF initialization menu" , "Back" , "bB");
  menu_add_item(menu , "Initalize all members/all parameters" , "1" , enkf_ui_init1 , enkf_main , NULL);
  menu_add_item(menu , "Initalize all members/one  parameter" , "2" , enkf_ui_init2 , enkf_main , NULL);
  menu_add_item(menu , "Initialize one member/all parameters" , "3" , enkf_ui_init3 , enkf_main , NULL);
  menu_add_item(menu , "Initialize one member/one parameter"  , "4" , enkf_ui_init4 , enkf_main , NULL);
  
  menu_run(menu);
  menu_free(menu);

}
