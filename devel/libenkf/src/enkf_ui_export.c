#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <field.h>
#include <enkf_state.h>
#include <enkf_fs.h>
#include <enkf_ui_util.h>
#include <field_config.h>



void enkf_ui_export_field(const enkf_main_type * enkf_main , field_file_format_type file_type) {
  const bool output_transform = true;
  const int prompt_len = 35;
  const enkf_config_node_type * config_node;
  state_enum analysis_state;
  char       analysis_state_char;
  int        iens , report_step;
  
  config_node = enkf_ui_util_scanf_parameter(enkf_main_get_ensemble_config(enkf_main) , prompt_len , false , FIELD  , invalid , &report_step , &analysis_state , &iens);
  if (analysis_state == analyzed)
    analysis_state_char = 'A';
  else
    analysis_state_char = 'F';
  

  {
    enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
    enkf_node_type * node = enkf_node_alloc(config_node);

    if (enkf_fs_has_node(fs , config_node , report_step , iens , analysis_state)) {
      char * filename = enkf_util_scanf_alloc_filename("File to store field in =>" , AUTO_MKDIR);
      enkf_fs_fread_node(fs , node , report_step , iens , analysis_state);
      {
	const field_type     * field = enkf_node_value_ptr(node);
	field_export(field , filename , NULL , file_type , output_transform);
      }
      free(filename);
    } else 
      printf("** Sorry node:%s does not exist for report step: %d%c.\n",enkf_config_node_get_key(config_node),report_step,analysis_state_char);
    
    enkf_node_free(node);
  } 
}


void enkf_ui_export_grdecl(void * enkf_main) {
  enkf_ui_export_field(enkf_main , ecl_grdecl_file);
}



void enkf_ui_export_roff(void * enkf_main) {
  enkf_ui_export_field(enkf_main , rms_roff_file);
}


void enkf_ui_export_restart_active(void * enkf_main) {
  enkf_ui_export_field(enkf_main , ecl_kw_file_active_cells);
}


void enkf_ui_export_restart_all(void * enkf_main) {
  enkf_ui_export_field(enkf_main , ecl_kw_file_all_cells);
}



void enkf_ui_export_profile(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 60;
    const int ens_size   = ensemble_config_get_size(ensemble_config);
    int iens1 , iens2;
    const int last_report = enkf_main_get_total_length( enkf_main );
    bool * iens_active  ;
    bool * report_active;

    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        direction;  /* 0: i running, 1: j running, 2: k running */
    int        total_cells;
    int       *cell_list; 
    path_fmt_type * file_fmt;

    config_node    = enkf_ui_util_scanf_parameter(ensemble_config , prompt_len , false , FIELD , invalid , NULL , &analysis_state , NULL);
    iens_active    = enkf_ui_util_scanf_alloc_iens_active( ens_size , prompt_len , &iens1 , &iens2); /* Not used yet ... */
    report_active  = enkf_ui_util_scanf_alloc_report_active( last_report , prompt_len );
    direction      = util_scanf_int_with_limits("Give scan direction 0:i  1:j  2:k" , prompt_len , 0 , 2);
    
    {
      const field_config_type * field_config = enkf_config_node_get_ref( config_node );
      int nx,ny,nz;
      int i1,i2,j1,j2,k1,k2;
      field_config_get_dims( field_config , &nx , &ny , &nz);
      i2 = j2 = k2 = 0;  /* Dummy for compiler */

      /* i1,i2,j1,j2,k1 and k2 should be incluseive */
      switch (direction) {
      case(0):
	i1 = 0; i2 = nx-1;
	enkf_ui_util_scanf_ijk__(field_config , prompt_len , NULL , &j1 , &k1);
	j2 = j1;
	k2 = k1;
	break;
      case(1):
	j1 = 0; j2 = ny-1;
	enkf_ui_util_scanf_ijk__(field_config , prompt_len , &i1 , NULL , &k1);
	i2 = i1; 
	k2 = k1;
	break;
      case(2):
	k1 = 0; k2 = nz-1;
	enkf_ui_util_scanf_ijk__(field_config , prompt_len , &i1 , &j1 , NULL);
	i2 = i1;
	j2 = j1;
	break;
      default:
	util_abort("%s: internal error \n",__func__);
      }
      total_cells = (i2 - i1 + 1) * (j2 - j1 + 1) * (k2 - k1 + 1);
      cell_list = util_malloc(total_cells * sizeof * cell_list , __func__);
      {
	int cell_index = 0;
	int i,j,k;
	for (i=i1; i<=i2; i++)
	  for (j= j1; j <=j2; j++)
	    for (k=k1; k <= k2; k++) {
	      cell_list[cell_index] = field_config_active_index(field_config , i,j,k);
	      cell_index++;
	    }
      }
    
      file_fmt = path_fmt_scanf_alloc("Give filename to store profiles (with TWO %d specifiers) =>" , 0 , NULL , false);
      {
	double * profile      = util_malloc(total_cells * sizeof * profile , __func__);
	int iens , report_step;
	enkf_node_type * node = enkf_node_alloc( config_node );
	enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
	
	for (report_step = 0; report_step <= last_report; report_step++) {
	  if (report_active[report_step]) {
	    for (iens = iens1; iens <= iens2; iens++) {
	      if (enkf_fs_has_node(fs , config_node , report_step , iens , analysis_state)) {
		enkf_fs_fread_node(fs , node , report_step , iens , analysis_state);
		{
		  const field_type * field = enkf_node_value_ptr( node );
		  int field_index;
		  for (field_index = 0 ; field_index < total_cells; field_index++)
		    profile[field_index] = field_iget_double(field , cell_list[field_index]);
		  {
		    char * filename = path_fmt_alloc_file(file_fmt , true , report_step , iens);
		    FILE * stream = util_fopen(filename , "w");
		    for (field_index = 0; field_index < total_cells; field_index++)
		      fprintf(stream, "%d  %g\n",field_index , profile[field_index]);
		    
		    fclose(stream);
		    free(filename);
		  }		    	      
		}
	      } else 
		fprintf(stderr," ** Warning field:%s is missing for member,report: %d,%d \n",enkf_config_node_get_key(config_node) , iens , report_step);
	    }
	  }
	}
	free(iens_active);
	free(report_active);
	free(profile);
      }
    }
    free(cell_list);
  }
}






void enkf_ui_export_cell(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 35;
    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        cell_nr;

    config_node = enkf_ui_util_scanf_parameter(ensemble_config , prompt_len , false , FIELD , invalid, NULL , &analysis_state , NULL);
    cell_nr = enkf_ui_util_scanf_ijk(enkf_config_node_get_ref(config_node) , prompt_len);
    {
      const int ens_size    = ensemble_config_get_size(ensemble_config);
      const int last_report = enkf_main_get_total_length( enkf_main );
      int iens1 , iens2;   
      bool * iens_active    = enkf_ui_util_scanf_alloc_iens_active( ens_size , prompt_len , &iens1 , &iens2); /* Not used yet ... */
      bool * report_active  = enkf_ui_util_scanf_alloc_report_active( last_report , prompt_len);
      double * cell_data    = util_malloc(ens_size * sizeof * cell_data , __func__);
      int iens , report_step; /* Observe that iens and report_step loops below should be inclusive.*/
      enkf_node_type * node = enkf_node_alloc( config_node );
      enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
      path_fmt_type * file_fmt = path_fmt_scanf_alloc("Give filename to store historgrams (with %d for report step) =>" , 0 , NULL , false);
	  
      
      for (report_step = 0; report_step <= last_report; report_step++) {
	if (report_active[report_step]) {
	  if (enkf_fs_has_node(fs , config_node , report_step , iens1 , analysis_state)) {
	    for (iens = iens1; iens <= iens2; iens++) {
	      if (enkf_fs_has_node(fs , config_node , report_step , iens , analysis_state)) {
		enkf_fs_fread_node(fs , node , report_step , iens , analysis_state);
		{
		  const field_type * field = enkf_node_value_ptr( node );
		  cell_data[iens] = field_iget_double(field , cell_nr);
		}
	      } else {
		fprintf(stderr," ** Warning field:%s is missing for member,report: %d,%d \n",enkf_config_node_get_key(config_node) , iens , report_step);
		cell_data[iens] = -1;
	      }
	    }
	    {
	      char * filename = path_fmt_alloc_file(file_fmt , true , report_step);
	      FILE * stream = util_fopen(filename , "w");
	      for (iens = iens1; iens <= iens2; iens++)
		fprintf(stream,"%g\n",cell_data[iens]);
	      
	      fclose(stream);
	      free(filename);
	    }
	  } else printf("Skipping report_step:%d \n",report_step);
	}
      }
      free(iens_active);
      free(report_active);
      free(cell_data);
      path_fmt_free(file_fmt);
    }
  }
}




void enkf_ui_export_time(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 35;
    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        cell_nr;
    
    config_node = enkf_ui_util_scanf_parameter(ensemble_config , prompt_len , true , FIELD ,invalid ,  NULL , &analysis_state , NULL);
    cell_nr = enkf_ui_util_scanf_ijk(enkf_config_node_get_ref(config_node) , prompt_len);
    {
      const int last_report = enkf_main_get_total_length( enkf_main );
      const int step1       = util_scanf_int_with_limits("First report step",prompt_len , 0 , last_report);
      const int step2       = util_scanf_int_with_limits("Last report step",prompt_len , step1 , last_report);
      const int ens_size    = ensemble_config_get_size(ensemble_config);
      int iens1 , iens2;   
      bool * iens_active    = enkf_ui_util_scanf_alloc_iens_active( ens_size , prompt_len , &iens1 , &iens2); /* Not used yet ... */
      double * x, *y;
      int iens; /* Observe that iens and report_step loops below should be inclusive.*/
      enkf_node_type * node = enkf_node_alloc( config_node );
      enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
      path_fmt_type * file_fmt = path_fmt_scanf_alloc("Give filename to store line (with %d for report iens) =>" , 0 , NULL , false);
      
      
      if (analysis_state == both) {
	x = util_malloc( 2 * (step2 - step1 + 1) * sizeof * x, __func__);
	y = util_malloc( 2 * (step2 - step1 + 1) * sizeof * y, __func__);
      }	else {
	x = util_malloc( (step2 - step1 + 1) * sizeof * x, __func__);
	y = util_malloc( (step2 - step1 + 1) * sizeof * y, __func__);
      }
	
      
      for (iens = iens1; iens <= iens2; iens++) {
	enkf_ui_util_get_time(fs , config_node , node , analysis_state , cell_nr , step1 , step2 , iens , x ,y);
	{
	  char * filename = path_fmt_alloc_file(file_fmt , true , iens);
	  FILE * stream = util_fopen(filename , "w");
	  int    index  = 0;
	  int    report_step;
	  for (report_step = step1; report_step <= step2; report_step++) {
	    fprintf(stream , "%g  %g \n",x[index] , y[index]);
	    index++;
	    if (analysis_state == both) {
	      fprintf(stream , "%g  %g \n",x[index] , y[index]);
	      index++;
	    }
	  }
	  fclose(stream);
	  free(filename);
	}
      }
      free(iens_active);
      free(x);
      free(y);
      path_fmt_free(file_fmt);
    }
  }
}




void enkf_ui_export_menu(void * arg) {

  enkf_main_type * enkf_main = enkf_main_safe_cast(arg);
  menu_type * menu = menu_alloc("Export EnKF data to other formats" , "Back" , "bB");
  menu_add_item(menu , "Export fields to RMS Roff format"       		, "rR" , enkf_ui_export_roff   	       , enkf_main , NULL);
  menu_add_item(menu , "Export fields to ECLIPSE grdecl format" 		, "gG" , enkf_ui_export_grdecl 	       , enkf_main , NULL);
  menu_add_item(menu , "Export fields to ECLIPSE restart format (active cells)" , "aA" , enkf_ui_export_restart_active , enkf_main , NULL);
  menu_add_item(menu , "Export fields to ECLIPSE restart format (all cells)"    , "lL" , enkf_ui_export_restart_all    , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Export cell values to text file(s)"                  	, "cC" , enkf_ui_export_cell    , enkf_main , NULL);
  menu_add_item(menu , "Export line profile of a field to text file(s)"      	, "pP" , enkf_ui_export_profile , enkf_main , NULL);
  menu_add_item(menu , "Export time development in one cell to text file(s)" 	, "tT" , enkf_ui_export_time    , enkf_main , NULL);
  menu_run(menu);
  menu_free(menu);
}
