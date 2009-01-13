#include <stdlib.h>
#include <menu.h>
#include <enkf_ui_util.h>
#include <enkf_fs.h>
#include <arg_pack.h>
#include <util.h>


void enkf_ui_fs_lsdir(void * arg) {
  enkf_fs_type    * fs      = (enkf_fs_type * ) arg;
  stringlist_type * dirlist = enkf_fs_alloc_dirlist( fs );
  int idir;

  printf("Avaliable enkf directories: ");
  for (idir = 0; idir < stringlist_get_size( dirlist ); idir++)
    printf("%s ",stringlist_iget( dirlist , idir ));
  
  printf("\n");
  stringlist_free( dirlist );
}


void enkf_ui_fs_menu(void * arg) {
  
   enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
   enkf_fs_type    * fs         = enkf_main_get_fs( enkf_main );

   menu_type * menu = menu_alloc("EnKF filesystem menu" , "Back" , "bB");
   menu_add_item(menu , "List available directories" , "lL" , enkf_ui_fs_lsdir , fs , NULL);
   
   {
     char * menu_label = util_alloc_sprintf("Set new directory for reading:%s" , enkf_fs_get_read_dir( fs ));
     arg_pack_type * arg_pack = arg_pack_alloc();
     arg_pack_append_ptr(arg_pack  , fs);
     arg_pack_append_bool(arg_pack , true);
     arg_pack_append_ptr(arg_pack  , menu_add_item(menu , menu_label , "rR" , enkf_fs_interactive_select_directory , arg_pack , arg_pack_free__));
     
     free(menu_label);
   }

   {
     char * menu_label = util_alloc_sprintf("Set new directory for writing:%s" , enkf_fs_get_write_dir( fs ));
     arg_pack_type * arg_pack = arg_pack_alloc();
     arg_pack_append_ptr(arg_pack  , fs);
     arg_pack_append_bool(arg_pack , false);
     arg_pack_append_ptr(arg_pack  , menu_add_item(menu , menu_label , "wW" , enkf_fs_interactive_select_directory , arg_pack , arg_pack_free__));
     
     free(menu_label);
   }

   menu_run(menu);
   menu_free(menu);
}

