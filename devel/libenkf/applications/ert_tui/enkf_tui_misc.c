#include <menu.h>
#include <util.h>
#include <enkf_types.h>
#include <enkf_main.h>
#include <enkf_state.h>
#include <enkf_tui_misc.h>




static void enkf_tui_misc_printf_subst_list(void * arg) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );

  /* These could/should be user input ... */
  int step1 = 0;  /* < 0 => no reinitializtion of the dynamic substitutions. */
  int step2 = 10;
  int iens  = 0;
  
  enkf_state_type * enkf_state = enkf_main_iget_state( enkf_main , iens );
  enkf_state_printf_subst_list( enkf_state , step1 , step2 );
}


void enkf_tui_misc_menu( void * arg) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );
  menu_type       * menu       = menu_alloc( "Misceallanous stuff" , "Back" , "bB");
  menu_add_item(menu , "List all \'magic\' <...> strings" , "lL" , enkf_tui_misc_printf_subst_list , enkf_main , NULL); 
  menu_run(menu);
  menu_free(menu);
}
