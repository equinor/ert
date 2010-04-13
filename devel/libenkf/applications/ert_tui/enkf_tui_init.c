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
#include <enkf_tui_util.h>
#include <enkf_tui_init.h>
#include <enkf_state.h>
#include <enkf_node.h>
#include <enkf_fs.h>
#include <msg.h>
#include <ensemble_config.h>




void enkf_tui_init(enkf_main_type * enkf_main, bool all_members , bool all_parameters) {
  const int prompt_len                         = 35;
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  int   ens_size                               = enkf_main_get_ensemble_size( enkf_main );
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
    stringlist_type * tmp_key_list = ensemble_config_alloc_keylist( ensemble_config);
    int num_keys = stringlist_get_size(tmp_key_list);
    for (int i = 0; i < num_keys; i++) {
      if (ensemble_config_var_type(ensemble_config , stringlist_iget(tmp_key_list,i)) != STATIC_STATE)
	stringlist_append_copy(param_list , stringlist_iget(tmp_key_list,i ));
    }
    stringlist_free(tmp_key_list);
  } else 
    stringlist_append_copy( param_list , enkf_config_node_get_key(enkf_tui_util_scanf_key(ensemble_config , prompt_len , INVALID , INVALID_VAR)) );

    

  enkf_main_initialize(enkf_main , param_list , iens1 , iens2);
  stringlist_free( param_list );
}


static void enkf_tui_init1(void * enkf_main) {
  enkf_tui_init(enkf_main, true , true);
}

static void enkf_tui_init2(void * enkf_main) {
  enkf_tui_init(enkf_main , true , false);
}

static void enkf_tui_init3(void * enkf_main) {
  enkf_tui_init(enkf_main , false , true );
}

static void enkf_tui_init4(void * enkf_main) {
  enkf_tui_init(enkf_main , false , false);
}



void enkf_tui_init_menu(void * arg) {
  enkf_main_type * enkf_main = enkf_main_safe_cast(arg);

  menu_type * menu = menu_alloc("Initialize from scratch" , "Back" , "bB");
  menu_add_item(menu , "Initialize all members/all parameters" , "1" , enkf_tui_init1 , enkf_main , NULL);
  menu_add_item(menu , "Initialize all members/one  parameter" , "2" , enkf_tui_init2 , enkf_main , NULL);
  menu_add_item(menu , "Initialize one member/all parameters" , "3" , enkf_tui_init3 , enkf_main , NULL);
  menu_add_item(menu , "Initialize one member/one parameter"  , "4" , enkf_tui_init4 , enkf_main , NULL);
  
  menu_run(menu);
  menu_free(menu);

}
