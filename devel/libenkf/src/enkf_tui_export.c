#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <field.h>
#include <field_config.h>
#include <enkf_state.h>
#include <enkf_fs.h>
#include <enkf_tui_util.h>
#include <field_config.h>
#include <msg.h>
#include <gen_data.h>



void enkf_tui_export_field(const enkf_main_type * enkf_main , field_file_format_type file_type) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const bool output_transform = true;
  const int prompt_len = 45;
  const enkf_config_node_type * config_node;
  state_enum analysis_state;
  const int last_report = enkf_main_get_total_length( enkf_main );
  int        iens1 , iens2 , iens , report_step;
  path_fmt_type * export_path;
  
  analysis_state = ANALYZED;  /* Hardcoded analyzed */
  config_node    = enkf_tui_util_scanf_key(enkf_main_get_ensemble_config(enkf_main) , prompt_len ,  FIELD  , INVALID_VAR );

  report_step = util_scanf_int_with_limits("Report step: ", prompt_len , 0 , last_report);
  enkf_tui_util_scanf_iens_range("Realizations members to export(0 - %d)" , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
  {
    char * path_fmt;
    util_printf_prompt("Filename to store files in (with %d) in: " , prompt_len , '=' , "=> ");
    path_fmt = util_alloc_stdin_line();
    export_path = path_fmt_alloc_path_fmt( path_fmt );
    free( path_fmt );
  }
  
  {
    enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
    enkf_node_type * node = enkf_node_alloc(config_node);

    for (iens = iens1; iens <= iens2; iens++) {
      if (enkf_fs_try_fread_node(fs , node , report_step , iens , BOTH)) {
	char * filename = path_fmt_alloc_path( export_path , false , iens);
	{
	  char * path;
	  util_alloc_file_components(filename , &path , NULL , NULL);
          if (path != NULL) {
            util_make_path( path );
            free( path );
          }
	}

	{
	  const field_type * field = enkf_node_value_ptr(node);
	  field_export(field , filename , NULL , file_type , output_transform);
	}
	free(filename);
      } else 
	printf("Warning: could not load realization:%d \n", iens);
    } 
    enkf_node_free(node);
  } 
}


void enkf_tui_export_grdecl(void * enkf_main) {
  enkf_tui_export_field(enkf_main , ECL_GRDECL_FILE);
}



void enkf_tui_export_roff(void * enkf_main) {
  enkf_tui_export_field(enkf_main , RMS_ROFF_FILE);
}


void enkf_tui_export_restart_active(void * enkf_main) {
  enkf_tui_export_field(enkf_main , ECL_KW_FILE_ACTIVE_CELLS);
}


void enkf_tui_export_restart_all(void * enkf_main) {
  enkf_tui_export_field(enkf_main , ECL_KW_FILE_ALL_CELLS);
}


void enkf_tui_export_gen_data(void * arg) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    enkf_var_type var_type;
    const int prompt_len = 60;
    int report_step;
    int iens1 , iens2;
    const int last_report = enkf_main_get_total_length( enkf_main );

    const enkf_config_node_type * config_node;
    state_enum state = ANALYZED;
    path_fmt_type * file_fmt;

    config_node    = enkf_tui_util_scanf_key(ensemble_config , prompt_len ,  GEN_DATA , INVALID_VAR);
    var_type       = enkf_config_node_get_var_type(config_node);
    if ((var_type == DYNAMIC_STATE) || (var_type == DYNAMIC_RESULT)) 
      state = enkf_tui_util_scanf_state("Plot Forecast/Analyzed: [F|A]" , prompt_len , false);
    else if (var_type == PARAMETER)
      state = ANALYZED;
    else 
      util_abort("%s: internal error \n",__func__);
    
    
    report_step = util_scanf_int_with_limits("Report step: ", prompt_len , 0 , last_report);
    enkf_tui_util_scanf_iens_range("Realizations members to export(0 - %d)" , ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
    {
      char path_fmt[512];
      util_printf_prompt("Filename to store files in (with %d) in: " , prompt_len , '=' , "=> ");
      scanf("%s" , path_fmt);
      file_fmt = path_fmt_alloc_path_fmt( path_fmt );
    }
    
    {
      enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
      enkf_node_type * node = enkf_node_alloc(config_node);
      int iens;

      for (iens = iens1; iens <= iens2; iens++) {
	if (enkf_fs_try_fread_node(fs , node , report_step , iens , state)) {
	  char * full_path = path_fmt_alloc_path( file_fmt , false , iens);
	  char * path;
	  char * ext;
	  char * basename;
	  util_alloc_file_components(full_path , &path , &basename , &ext);
	  if (path != NULL) util_make_path( path );
	  
	  {
	    const gen_data_type * gen_data = enkf_node_value_ptr(node);
	    char * file_with_ext = util_alloc_filename(NULL , basename , ext);
	    gen_data_ecl_write(gen_data , path , file_with_ext , NULL);
	    free(file_with_ext);
	  }
	  
	  free(full_path);
	  util_safe_free(path);
	  util_safe_free(ext);
	  util_safe_free(basename);
	}
      } 
      enkf_node_free(node);
    } 
  }
}



void enkf_tui_export_profile(void * enkf_main) {
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

    analysis_state = ANALYZED; /* */
    config_node    = enkf_tui_util_scanf_key(ensemble_config , prompt_len ,  FIELD , INVALID_VAR);
    iens_active    = enkf_tui_util_scanf_alloc_iens_active( ens_size , prompt_len , &iens1 , &iens2); /* Not used yet ... */
    report_active  = enkf_tui_util_scanf_alloc_report_active( last_report , prompt_len );
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
	enkf_tui_util_scanf_ijk__(field_config , prompt_len , NULL , &j1 , &k1);
	j2 = j1;
	k2 = k1;
	break;
      case(1):
	j1 = 0; j2 = ny-1;
	enkf_tui_util_scanf_ijk__(field_config , prompt_len , &i1 , NULL , &k1);
	i2 = i1; 
	k2 = k1;
	break;
      case(2):
	k1 = 0; k2 = nz-1;
	enkf_tui_util_scanf_ijk__(field_config , prompt_len , &i1 , &j1 , NULL);
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






void enkf_tui_export_cell(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 35;
    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        cell_nr;

    analysis_state = ANALYZED;
    config_node = enkf_tui_util_scanf_key(ensemble_config , prompt_len , FIELD , INVALID_VAR);
    cell_nr = enkf_tui_util_scanf_ijk(enkf_config_node_get_ref(config_node) , prompt_len);
    {
      const int ens_size    = ensemble_config_get_size(ensemble_config);
      const int last_report = enkf_main_get_total_length( enkf_main );
      int iens1 , iens2;   
      bool * iens_active    = enkf_tui_util_scanf_alloc_iens_active( ens_size , prompt_len , &iens1 , &iens2); /* Not used yet ... */
      bool * report_active  = enkf_tui_util_scanf_alloc_report_active( last_report , prompt_len);
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




void enkf_tui_export_time(void * enkf_main) {
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  {
    const int prompt_len = 35;
    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        cell_nr;
    
    analysis_state = ANALYZED;
    config_node = enkf_tui_util_scanf_key(ensemble_config , prompt_len , FIELD ,INVALID_VAR);
    cell_nr = enkf_tui_util_scanf_ijk(enkf_config_node_get_ref(config_node) , prompt_len);
    {
      const int last_report = enkf_main_get_total_length( enkf_main );
      const int step1       = util_scanf_int_with_limits("First report step",prompt_len , 0 , last_report);
      const int step2       = util_scanf_int_with_limits("Last report step",prompt_len , step1 , last_report);
      const int ens_size    = ensemble_config_get_size(ensemble_config);
      int iens1 , iens2;   
      bool * iens_active    = enkf_tui_util_scanf_alloc_iens_active( ens_size , prompt_len , &iens1 , &iens2); /* Not used yet ... */
      double * x, *y;
      int iens; /* Observe that iens and report_step loops below should be inclusive.*/
      enkf_node_type * node = enkf_node_alloc( config_node );
      enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
      path_fmt_type * file_fmt = path_fmt_scanf_alloc("Give filename to store line (with %d for report iens) =>" , 0 , NULL , false);
      
      
      if (analysis_state == BOTH) {
	x = util_malloc( 2 * (step2 - step1 + 1) * sizeof * x, __func__);
	y = util_malloc( 2 * (step2 - step1 + 1) * sizeof * y, __func__);
      }	else {
	x = util_malloc( (step2 - step1 + 1) * sizeof * x, __func__);
	y = util_malloc( (step2 - step1 + 1) * sizeof * y, __func__);
      }
	
      
      for (iens = iens1; iens <= iens2; iens++) {
	enkf_tui_util_get_time(fs , config_node , node , analysis_state , cell_nr , step1 , step2 , iens , x ,y);
	{
	  char * filename = path_fmt_alloc_file(file_fmt , true , iens);
	  FILE * stream = util_fopen(filename , "w");
	  int    index  = 0;
	  int    report_step;
	  for (report_step = step1; report_step <= step2; report_step++) {
	    fprintf(stream , "%g  %g \n",x[index] , y[index]);
	    index++;
	    if (analysis_state == BOTH) {
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


void enkf_tui_export_python_module(void * arg ) {
  enkf_main_type * enkf_main                   = enkf_main_safe_cast( arg ); 
  const int prompt_len                         = 45;
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size                           = ensemble_config_get_size( ensemble_config );
  enkf_fs_type   * fs                          = enkf_main_get_fs(enkf_main);
  char ** kw_list;
  char * keyword_string;
  char * step_string;
  char * module_name;
  char * module_file;
  int  * step_list;
  int    num_step , num_kw;
  
  util_printf_prompt("Keywords to export" , prompt_len , '=' , "=> ");
  keyword_string = util_alloc_stdin_line();
  util_printf_prompt("Timesteps to export" , prompt_len , '=' , "=> ");
  step_string    = util_alloc_stdin_line();
  util_printf_prompt("Name of python module" , prompt_len , '=' , "=> ");
  module_name    = util_alloc_stdin_line();
  module_file    = util_alloc_sprintf("%s.py" , module_name );
  step_list      = util_sscanf_alloc_active_list( step_string , &num_step );
  util_split_string( keyword_string , " " , &num_kw , &kw_list);
  {
    FILE * stream = util_fopen(module_file , "w");
    int ikw;
    fprintf(stream , "data = [");
    for (ikw = 0; ikw < num_kw; ikw++) {
      char * index_key;
      const enkf_config_node_type * config_node = ensemble_config_user_get_node( ensemble_config , kw_list[ikw] , &index_key);
      if (config_node == NULL) 
	fprintf(stderr,"Warning: could not locate node: %s \n", kw_list[ikw]);
      else {
	enkf_node_type * node = enkf_node_alloc( config_node );
	for (int istep = 0; istep < num_step; istep++) {
	  bool valid;
	  int  step = step_list[istep];
	  fprintf(stream , "(\"%s\" , %d , [" , kw_list[ikw] , step);
	  for (int iens = 0; iens < ens_size; iens++) {
	    enkf_fs_fread_node(fs , node , step , iens , FORECAST);
	    fprintf(stream , "%g " , enkf_node_user_get( node , index_key , &valid));
	    if (iens < (ens_size -1 ))
	      fprintf(stream , ",");
	    else
	      fprintf(stream , "]");
	  }
	  if ((istep == (num_step - 1)) && (ikw == (num_kw - 1)))
	    fprintf(stream , ")]");
	  else
	    fprintf(stream , "),\n");
	}
	free(index_key);
	enkf_node_free( node );
      }
    }
    fclose(stream);
  }
  

  util_free_stringlist( kw_list , num_kw );
  free(module_name);
  free(module_file);
  free( step_list );
  free( step_string );
  free( keyword_string );
}

/*****************************************************************/

void enkf_tui_export_fieldP(void * arg) {
  enkf_main_type * enkf_main                   = enkf_main_safe_cast( arg ); 
  const int prompt_len                         = 45;
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  state_enum analysis_state   	      	       = BOTH;
  const enkf_config_node_type * config_node    = enkf_tui_util_scanf_key(ensemble_config , prompt_len ,  FIELD  , INVALID_VAR );
  int iens1                   	      	       = 0;
  int iens2                   	      	       = ensemble_config_get_size(ensemble_config);
  const int last_report                        = enkf_main_get_total_length( enkf_main );
  int report_step                              = util_scanf_int_with_limits("Report step: ", prompt_len , 0 , last_report);
  double lower_limit                           = util_scanf_double("Lower limit", prompt_len);
  double upper_limit                           = util_scanf_double("Upper limit", prompt_len);
  char * export_file;
  util_printf_prompt("Filename to store file: " , prompt_len , '=' , "=> ");
  export_file = util_alloc_stdin_line();
  {
    enkf_fs_type   * fs        = enkf_main_get_fs(enkf_main);
    enkf_node_type ** ensemble = enkf_fs_fread_alloc_ensemble( fs , config_node , report_step , iens1 , iens2 , analysis_state );
    enkf_node_type *  sum      = enkf_node_alloc( config_node );
    int active_ens_size        = 0;
    int iens;
    
    enkf_node_clear( sum );
    {
      /* OK going low level */
      field_type * sum_field = enkf_node_value_ptr( sum );

      for (iens = iens1; iens < iens2; iens++) {
        if (ensemble[iens - iens1] != NULL) {
          field_type * field     = enkf_node_value_ptr( ensemble[iens - iens1] );
          field_update_sum( sum_field , field , lower_limit , upper_limit);
          active_ens_size++;
        }
      }
      if (active_ens_size > 0) {
        field_scale( sum_field , 1.0 / active_ens_size );
        {
          char * path;
          util_alloc_file_components( export_file , &path , NULL , NULL);
          if (path != NULL) {
            util_make_path( path );
            free( path );
          }
        }
        field_export(sum_field , export_file , NULL , RMS_ROFF_FILE , false);
      } else fprintf(stderr,"Warning: no data found \n");
    }    
    
    for (iens = iens1; iens < iens2; iens++) {
      if (ensemble[iens - iens1] != NULL)
        enkf_node_free( ensemble[iens - iens1] );
    }

    free( ensemble );
    enkf_node_free( sum );
  }
  free( export_file );
}


/*****************************************************************/


/**
   This is a very simple function for exporting a scalar value for all
   member/report steps to a CSV file. The file is characterized by:

    * Missing elements are represented with an empty string.

    * The header strings are quoted with "".

    * End of line is \r\n

   Unfortunately Excel does not seem to recognize the csv format, and
   it is necessary to go through a text import wizard in excel. To
   import this file you go through the following hoops in excel:

    1. [Data> - [Import external data> - [Import data>

    2. Select the file to import from.

    3. The text import wizard from excel should pop up:

         1. Select (*) Delimited - press next.
         2. Select delimiter "Comma" - press next.
         3. press finish.
  
      Finally you are asked where in the excel workbook you want to
      insert the data.
*/


#define CSV_NEWLINE        "\r\n"
#define CSV_MISSING_VALUE  ""
#define CSV_SEP            ","


void enkf_tui_export_scalar2csv(void * arg) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( arg );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const enkf_config_node_type * config_node;
  const int prompt_len  = 60;
  char * user_key, *key_index;
  
  util_printf_prompt("Scalar to export (KEY:INDEX)" , prompt_len , '=' , "=> "); user_key = util_alloc_stdin_line();
  config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
  if (config_node != NULL) {
    int    report_step , first_report, last_report;
    int    iens1 , iens2, iens;
    char * csv_file;
    
    iens2 	 = ensemble_config_get_size(ensemble_config) - 1;
    iens1 	 = 0;   
    first_report = 0;
    last_report  = enkf_main_get_total_length( enkf_main );
    {
      char * path;
      char * prompt = util_alloc_sprintf("File to store \'%s\'", user_key);
      util_printf_prompt(prompt , prompt_len , '=' , "=> ");
      csv_file = util_alloc_stdin_line();

      util_alloc_file_components( csv_file , &path , NULL , NULL);
      if (path != NULL) {
	if (util_path_exists( path )) {
	  if (!util_is_directory( path )) {
	    /* The path component already exists in the filesystem - and it is not a directory - we leave the building. */
	    fprintf(stderr,"Sorry: %s already exists - and is not a directory.\n",path);
	    free(path);
	    free(csv_file);
            free(user_key);
            return ;
	  }
	} else {
	  /* The path does not exist - we make it. */
	  enkf_tui_util_msg("Creating new directory: %s\n" , path);
	  util_make_path( path );
	}
      }
      free(prompt);
    }
    {
      /* Seriously manual creation of csv file. */
      enkf_fs_type * fs     = enkf_main_get_fs(enkf_main);
      enkf_node_type * node = enkf_node_alloc( config_node );
      FILE * stream         = util_fopen( csv_file , "w");
      msg_type * msg        = msg_alloc("Exporting report_step/member: ");
      
      /* Header line */
      fprintf(stream , "\"Report step\"");
      for (iens = iens1; iens <= iens2; iens++) 
	fprintf(stream , "%s\"%s(%d)\"" , CSV_SEP , user_key , iens);
      fprintf(stream , CSV_NEWLINE);
      
      msg_show(msg);
      for (report_step = first_report; report_step <= last_report; report_step++) {
	fprintf(stream , "%6d" , report_step);
	for (iens = iens1; iens <= iens2; iens++) {
	  char label[32];
	  /* 
	     Have not implemented a choice on forecast/analyzed. Tries
	     analyzed first, then forecast.
	  */
	  sprintf(label , "%03d/%03d" , report_step , iens);
	  msg_update( msg , label);
	  if (enkf_fs_try_fread_node(fs , node , report_step , iens , BOTH)) {
	    bool   valid;
	    double value = enkf_node_user_get( node , key_index , &valid);
	    if (valid)
	      fprintf(stream , "%s%g" , CSV_SEP , value);
	    else
	      fprintf(stream , "%s%s" , CSV_SEP , CSV_MISSING_VALUE);
	  } else
	    fprintf(stream , "%s%s" , CSV_SEP , CSV_MISSING_VALUE);
	}
	fprintf(stream , CSV_NEWLINE);
      }

      msg_free( msg , true );
      enkf_node_free( node );
      fclose(stream);
    }
  } else 
    fprintf(stderr,"Sorry - could not find any nodes with key:%s\n",user_key);
  
  free(user_key);
}


#undef CSV_NEWLINE        
#undef CSV_MISSING_VALUE  
#undef CSV_SEP




void enkf_tui_export_menu(void * arg) {

  enkf_main_type * enkf_main = enkf_main_safe_cast(arg);
  menu_type * menu = menu_alloc("Export data to other formats" , "Back" , "bB");
  menu_add_item(menu , "Export scalar value to CSV file"                        , "xX" , enkf_tui_export_scalar2csv     , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Export fields to RMS Roff format"       		, "rR" , enkf_tui_export_roff   	       , enkf_main , NULL);
  menu_add_item(menu , "Export fields to ECLIPSE grdecl format" 		, "gG" , enkf_tui_export_grdecl 	       , enkf_main , NULL);
  menu_add_item(menu , "Export fields to ECLIPSE restart format (active cells)" , "aA" , enkf_tui_export_restart_active , enkf_main , NULL);
  menu_add_item(menu , "Export fields to ECLIPSE restart format (all cells)"    , "lL" , enkf_tui_export_restart_all    , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Export P( a =< x < b )" , "sS" , enkf_tui_export_fieldP , enkf_main , NULL);                 
  menu_add_separator(menu);
  menu_add_item(menu , "Export Python module of" , "yY"  , enkf_tui_export_python_module , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Export cell values to text file(s)"                  	, "cC" , enkf_tui_export_cell    , enkf_main , NULL);
  menu_add_item(menu , "Export line profile of a field to text file(s)"      	, "pP" , enkf_tui_export_profile , enkf_main , NULL);
  menu_add_item(menu , "Export time development in one cell to text file(s)" 	, "tT" , enkf_tui_export_time    , enkf_main , NULL);
  menu_add_separator(menu);
  menu_add_item(menu , "Export GEN_DATA/GEN_PARAM to file"                      , "dD" , enkf_tui_export_gen_data , enkf_main , NULL);
  menu_run(menu);
  menu_free(menu);
}
