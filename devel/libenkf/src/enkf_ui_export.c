#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <void_arg.h>
#include <void_arg.h>
#include <field.h>
#include <enkf_state.h>
#include <enkf_fs.h>
#include <enkf_ui_util.h>
#include <field_config.h>



void enkf_ui_export_field(const enkf_main_type * enkf_main , field_file_format_type file_type) {
  char * key;
  const enkf_config_node_type * config_node;
  enkf_node_type * node;
  state_enum analysis_state;
  char       analysis_state_char;
  int        iens , report_step;
  
  do {
    enkf_ui_util_scanf_parameter(enkf_main_get_config(enkf_main) , &key , &report_step , &analysis_state , &iens);
    if (analysis_state == analyzed)
      analysis_state_char = 'A';
    else
      analysis_state_char = 'F';
    
    config_node = enkf_main_get_config_node(enkf_main , key);
    if (enkf_config_node_get_impl_type(config_node) != FIELD) {
      printf("** EnKF parameter:%s is not a field - can not be exported this way.\n", key);
      free(key);
    }
  } while (enkf_config_node_get_impl_type(config_node) != FIELD);


  {
    enkf_fs_type   * fs   = enkf_main_get_fs_ref(enkf_main);
    enkf_node_type * node = enkf_node_alloc(config_node);

    if (enkf_fs_has_node(fs , node , report_step , iens , analysis_state)) {
      char * filename = enkf_util_scanf_alloc_filename("File to store field in =>" , AUTO_MKDIR);
      enkf_fs_fread_node(fs , node , report_step , iens , analysis_state);
      {
	const field_type     * field = enkf_node_value_ptr(node);
	field_export(field , filename , file_type);
      }
      free(filename);
    } else 
      printf("** Sorry node:%s does not exist for report step: %d%c.\n",key,report_step,analysis_state_char);

    enkf_node_free(node);
  } 
  
  free(key);
}


void enkf_ui_export_grdecl(void *_arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);
  
  enkf_ui_export_field(enkf_main , ecl_grdecl_file);
}



void enkf_ui_export_roff(void *_arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);
  
  enkf_ui_export_field(enkf_main , rms_roff_file);
}


void enkf_ui_export_restart_active(void *_arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);
  
  enkf_ui_export_field(enkf_main , ecl_kw_file_active_cells);
}


void enkf_ui_export_restart_all(void *_arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);
  
  enkf_ui_export_field(enkf_main , ecl_kw_file_all_cells);
}



void enkf_ui_export_cell(void *_arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);

  {
    char * key;
    const enkf_config_node_type * config_node;
    enkf_node_type * node;
    state_enum analysis_state;
    char       analysis_state_char;
    int        report_step;
    int        cell_nr;
    do {
      enkf_ui_util_scanf_parameter(enkf_main_get_config(enkf_main) , &key , NULL , &analysis_state , NULL);
      if (analysis_state == analyzed)
	analysis_state_char = 'A';
      else
	analysis_state_char = 'F';
      
      config_node = enkf_main_get_config_node(enkf_main , key);
      if (enkf_config_node_get_impl_type(config_node) != FIELD) {
	printf("** EnKF parameter:%s is not a field - can not be exported this way.\n", key);
	free(key);
      }
    } while (enkf_config_node_get_impl_type(config_node) != FIELD);  

    cell_nr = enkf_ui_util_scanf_ijk(enkf_config_node_get_ref(config_node));
    /*
    {
      int iens1, int iens2;
      int iens;
      enkf_node_type * node = enkf_node_alloc(key , config_node);
      enkf_fs_type   * fs   = enkf_main_get_fs_ref(enkf_main);

      
      for (iens = 0; iens < iens1; iens < iens2) {
	if (enkf_fs_has_node(fs , node , report_step , iens , analysis_state)) {
	  char * filename = enkf_util_scanf_alloc_filename("File to store field in =>" , AUTO_MKDIR);
	  enkf_fs_fread_node(fs , node , report_step , iens , analysis_state);
	}
      }
    }
    */
  }
}


void enkf_ui_export_menu(void * _arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);
  enkf_sched_type * enkf_sched = void_arg_get_ptr(arg , 1);

  
  menu_type * menu = menu_alloc("Export EnKF data to other formats" , "qQ");
  menu_add_item(menu , "Export fields to RMS Roff format"       , "rR" , enkf_ui_export_roff   , arg);
  menu_add_item(menu , "Export fields to ECLIPSE grdecl format" , "gG" , enkf_ui_export_grdecl , arg);
  menu_add_item(menu , "Export fields to ECLIPSE restart format (active cells)" , "aA" , enkf_ui_export_restart_active , arg);
  menu_add_item(menu , "Export fields to ECLIPSE restart format (all cells)"    , "lL" , enkf_ui_export_restart_all   , arg);
  menu_add_separator(menu);
  menu_add_item(menu , "Export cell values to text file(s)" , "cC" , enkf_ui_export_cell , arg);
  menu_run(menu);
  menu_free(menu);
}
