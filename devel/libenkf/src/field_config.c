#include <stdlib.h>
#include <string.h>
#include <enkf_types.h>
#include <util.h>
#include <field_config.h>
#include <enkf_macros.h>
#include <ecl_grid.h>
#include <ecl_kw.h>
#include <ecl_util.h>
#include <rms_file.h>
#include <rms_util.h>
#include <path_fmt.h>
#include <math.h>
#include <field_active.h>
#include <active_list.h>
#include <field_trans.h>
#include <field_common.h>  

/**
   About transformations and truncations
   -------------------------------------

   The values of the fields data can be automagically manipulated through two methods:

   * You can specify a min and a max value which will serve as truncation.

   * You can specify transformation functions which are applied to the field as follows:

     init_transform: This function is applied to the field when the
        field is loaded the first time, i.e. initialized. It is *NOT*
        applied under subsequent loads of dynamic fields during the
        execution.

     output_transform: This function is applied to the field before it
        is exported to eclipse.

     input_transform: This function is applied each time a field is
        loaded in from the forward model; i.e. this transformation
        applies to dynamic fields.



 							    _______________________________         ___
							   /   	       	 		   \	    /|\
                                                           | Forward model (i.e. ECLIPSE)  |	     |
                                                           | generates dynamic fields like |	     |
                                                           | PRESSURE and SATURATIONS	   |	     |
							   \_______________________________/	     |	   This code is run
							   		  |		   	     |	   every time a field
									  |			     |	   is loaded FROM the
									 \|/			     |	   forward model into
									  | 			     |	   EnKF.
								  ________|_________		     |
								 /     	       	    \		     |
								 | Input transform  |		     |
								 \__________________/		     |
								    	  |			     |
								    	  |			     |
								    	 \|/			     |
								    	  |			     |
                            	                          ________________|__________________	   _\|/_
_______________                       ___________	 /                                   \
               \                     /         	 \	 |  The internal representation      |
 Geo Modelling |                     | init-     |	 |  of the field. This (should)      |
 creates a     |==>===============>==| transform |===>===|  be a normally distributed        |
 realization   |                     | 	     	 |	 |  variable suitable for updates    |
_______________/                     \___________/	 |  with EnKF.                       |
                               	              		 \___________________________________/   ___
|<----   This path is ONLY executed during INIT ------->|                  |                     /|\
         Observe that there is no truncation                              \|/                     |
         on load.					          _________|__________		  |
                                                                 /                    \		  |   This code is run
                                                                 |  Output transform  |		  |   every time a field
                                                                 \____________________/		  |   is exported from
                                                                           |			  |   enkf to the forward
                                                                          \|/			  |   model - i.e. ECLIPSE.
							          _________|__________		  |
                                                                 /                    \		  |
                                                                 | Truncate min/max   | 	  |
                                                                 \____________________/		  |
                                                                           |			  |
                                                                          \|/			  |
							          _________|__________		  |
                                                                 /                    \		  |
                                                                 |    FORWARD MODEL   | 	  |
                                                                 \____________________/		_\|/_






*/

/*Observe the following convention:

    global_index:  [0 , nx*ny*nz)
    active_index:  [0 , nactive)
*/

#define FIELD_CONFIG_ID 78269

struct field_config_struct {
  CONFIG_STD_FIELDS;
  char                * ecl_kw_name;    /* Name/key ... */
  int nx,ny,nz;                         /* The number of elements in the three directions. */
  const ecl_grid_type * grid;           /* A shared reference to the grid this field is defined on. */
  bool  private_grid;
  
  active_list_type * active_list;

  int             	  truncation;           /* How the field should be trunacted before exporting for simulation, and for the inital import. */
  double   	  	  min_value;            /* The min value used in truncation. */
  double   	  	  max_value;            /* The maximum value used in truncation. */

  field_file_format_type  export_format;
  field_file_format_type  import_format;
  int             	  sizeof_ctype;
  ecl_type_enum           internal_ecl_type;
  ecl_type_enum           export_ecl_type;
  path_fmt_type         * init_file_fmt; /* The format for loding init_files - if this is NULL the initialization is done by the forward model. */

  bool __enkf_mode;                      /* See doc of functions field_config_set_key() / field_config_enkf_OFF() */
  bool fmt_file;
  bool write_compressed;
  bool add_perturbation;

  field_type              * min_std;
  field_func_type         * output_transform;     /* Function to apply to the data before they are exported - NULL: no transform. */
  field_func_type         * init_transform;       /* Function to apply on the data when they are loaded the first time - i.e. initialized. NULL : no transform*/
  field_func_type         * input_transform;      /* Function to apply on the data when they are loaded from the forward model - i.e. for dynamic data. */
};



/*****************************************************************/

void field_config_set_ecl_kw_name(field_config_type * config , const char * ecl_kw_name) {
  config->ecl_kw_name = util_realloc_string_copy(config->ecl_kw_name , ecl_kw_name);
}



void field_config_set_ecl_type(field_config_type * config , ecl_type_enum ecl_type) {
  config->internal_ecl_type     = ecl_type;
  config->sizeof_ctype = ecl_util_get_sizeof_ctype(ecl_type);
}



static const char * field_config_file_type_string(field_file_format_type file_type) {
  switch (file_type) {
  case(RMS_ROFF_FILE):
    return "Binary ROFF file from RMS";
    break;
  case(ECL_KW_FILE):
    return "ECLIPSE file in restart format";
    break;
  case(ECL_KW_FILE_ALL_CELLS):
    return "ECLIPSE file in restart format (all cells)";
    break;
  case(ECL_KW_FILE_ACTIVE_CELLS):
    return "ECLIPSE file in restart format (active cells)";
    break;
  case(ECL_GRDECL_FILE):
    return "ECLIPSE file in grdecl format";
    break;
  default:
    fprintf(stderr,"%s: invalid file type \n",__func__);
    abort();
  }
}



/**
   This function takes a field_file_format_type variable, and returns
   a string containing a default extension for files of this type. For
   ecl_kw_file it will return NULL, i.e. no default extension.

     rms_roff_file   => ROFF
     ecl_grdecl_file => GRDECL
     ecl_kw_file_xxx => NULL

   It will return UPPERCASE or lowercase depending on the value of the
   second argument.
*/


const char * field_config_default_extension(field_file_format_type file_type, bool upper_case) {
  if (file_type == RMS_ROFF_FILE) {
    if (upper_case)
      return "ROFF";
    else
      return "roff";
  } else if (file_type == ECL_GRDECL_FILE) {
    if (upper_case)
      return "GRDECL";
    else
      return "grdecl";
  } else
    return NULL;
}




static bool field_config_valid_file_type(field_file_format_type file_type, bool import) {
  if (import) {
    if (file_type == RMS_ROFF_FILE || file_type == ECL_KW_FILE || file_type == ECL_GRDECL_FILE)
      return true;
    else
      return false;
  } else {
    if (file_type == RMS_ROFF_FILE || file_type == ECL_KW_FILE_ACTIVE_CELLS || file_type == ECL_KW_FILE_ALL_CELLS || file_type == ECL_GRDECL_FILE)
      return true;
    else
      return false;
  }
}


static field_file_format_type field_config_default_export_format(const char * filename) {
  field_file_format_type export_format = ECL_KW_FILE_ALL_CELLS;   /* Suitable for PERMX/PORO/... */
  char * extension;
  util_alloc_file_components( filename , NULL , NULL , &extension);
  if (extension != NULL) {
    util_strupr(extension);

    if (strcmp(extension , "GRDECL") == 0)
      export_format = ECL_GRDECL_FILE;
    else if (strcmp(extension , "ROFF") == 0)
      export_format = RMS_ROFF_FILE;

    free(extension);
  }

  return export_format;
}




/**
   This function prompts the user for a file type.

   If the parameter 'import' is true we provide the alternative
   ecl_kw_file (in that case the program itself will determine
   whether) the file contains all cells (i.e. PERMX) or only active
   cells (i.e. pressure).

   If the parameter 'import' is false the user must specify whether we
   are considering all cells, or only active cells.
*/

field_file_format_type field_config_manual_file_type(const char * prompt , bool import) {
  int int_file_type;
  printf("\n%s\n",prompt);
  printf("----------------------------------------------------------------\n");
  printf(" %3d: %s.\n" , RMS_ROFF_FILE   , field_config_file_type_string(RMS_ROFF_FILE));
  if (import)
    printf(" %3d: %s.\n" , ECL_KW_FILE     , field_config_file_type_string(ECL_KW_FILE));
  else {
    printf(" %3d: %s.\n" , ECL_KW_FILE_ACTIVE_CELLS  , field_config_file_type_string(ECL_KW_FILE_ACTIVE_CELLS));
    printf(" %3d: %s.\n" , ECL_KW_FILE_ALL_CELLS     , field_config_file_type_string(ECL_KW_FILE_ALL_CELLS));
  }
  printf(" %3d: %s.\n" , ECL_GRDECL_FILE , field_config_file_type_string(ECL_GRDECL_FILE));
  printf("----------------------------------------------------------------\n");
  do {
    int_file_type = util_scanf_int("" , 2);
    if (!field_config_valid_file_type(int_file_type, import))
      int_file_type = UNDEFINED_FORMAT;
  } while(int_file_type == UNDEFINED_FORMAT);
  return int_file_type;
}




/**
This function takes in a filename and tries to guess the type of the
file. It can determine the following three types of files:

  ecl_kw_file: This is a file containg ecl_kw instances in the form found
     in eclipse restart files.

  rms_roff_file: An rms roff file - obviously.

  ecl_grdecl_file: This is a file containing a parameter of the form
     found in eclipse grid declaration files, i.e. formatted, one
     keyword and all elements (active and not).

  The latter test is the weakest. Observe that the function will
  happily return unkown_file if none of these types are recognized,
  i.e. it is *essential* to check the return value.

*/
field_file_format_type field_config_guess_file_type(const char * filename ) {
  bool fmt_file = util_fmt_bit8(filename );
  FILE * stream = util_fopen(filename , "r");

  field_file_format_type file_type;
  if (ecl_kw_is_kw_file(stream , fmt_file ))
    file_type = ECL_KW_FILE;
  else if (rms_file_is_roff(stream))
    file_type = RMS_ROFF_FILE;
  else if (ecl_kw_is_grdecl_file(stream))  /* This is the weakest test - and should be last in a cascading if / else hierarchy. */
    file_type = ECL_GRDECL_FILE;
  else
    file_type = UNDEFINED_FORMAT;              /* MUST Check on this return value */

  fclose(stream);
  return file_type;
}



field_type * field_config_get_min_std( const field_config_type * field_config ) {
  return field_config->min_std;
}  


field_file_format_type field_config_get_export_format(const field_config_type * field_config) {
  return field_config->export_format;
}

field_file_format_type field_config_get_import_format(const field_config_type * field_config) {
  return field_config->import_format;
}


void field_config_set_grid(field_config_type * config, const ecl_grid_type * grid , bool private_grid) {
  config->grid         = grid;
  config->private_grid = private_grid;
  ecl_grid_get_dims(grid , &config->nx , &config->ny , &config->nz , &config->data_size);
}


const char * field_config_get_grid_name( const field_config_type * config) {
  return ecl_grid_get_name( config->grid );
}


static field_config_type * field_config_alloc__(const char * ecl_kw_name 	      	   , /* 1: Keyword name */
						ecl_type_enum ecl_type   	      	   , /* 2: Type of underlying data.*/
						const ecl_grid_type * ecl_grid        	   , /* 3: The underlying grid */
						field_file_format_type import_format  	   , /* 4: The format used when loading instances of this field. */
						field_file_format_type export_format  	   , /* 5: The format used when exporting (for ECLIPSE) instance of this field. */
						field_trans_table_type * field_trans_table , /* 6: Table of available transformation functions for input/output. */
						const stringlist_type * options) {           /* 7: Extra options in format: MIN:0.001   MAX:0.89 ...  */
  
  field_config_type *config = util_malloc(sizeof *config, __func__);
  config->__type_id = FIELD_CONFIG_ID;
  /*
    Observe that size is the number of *ACTIVE* cells,
    and generally *not* equal to nx*ny*nz.
  */

  /*
     The import format should in general be undefined_format - then
     the type will be determined automagically (unless it is
     restart_block).
  */
  config->export_format 	   = export_format;
  config->import_format 	   = import_format;
  
  field_config_set_grid(config , ecl_grid , false);
  config->ecl_kw_name = NULL;
  field_config_set_ecl_kw_name(config , ecl_kw_name);
  field_config_set_ecl_type(config , ecl_type);

  config->truncation               = TRUNCATE_NONE;
  config->__enkf_mode              = true;
  config->fmt_file    	      	   = false;
  config->write_compressed    	   = true;
  config->init_file_fmt            = NULL;
  config->output_transform         = NULL;
  config->init_transform           = NULL;
  config->input_transform          = NULL;
  config->active_list              = active_list_alloc( ALL_ACTIVE );
  config->min_std                  = NULL;

  /* Starting on the options. */
  {
    hash_type * opt_hash = hash_alloc_from_options( options );
    hash_iter_type * iter = hash_iter_alloc(opt_hash);
    const char * option = hash_iter_get_next_key(iter);
    const char * min_std_file = NULL;
    while (option != NULL) {
      const char * value = hash_get( opt_hash , option );
      bool option_OK     = false;

      /*
	This could (should ??) have been implemented with
	if-then-else; isolated if-blocks have been chosen for
	clarity. Must update option_OK in every block, and check it
	at the bottom.

      */

      if (strcmp(option , "MIN") == 0) {
	double min_value;
	if (util_sscanf_double( value , &min_value)) {
	  config->min_value  = min_value;
	  util_bitmask_on( &config->truncation , TRUNCATE_MIN );
	} else
	  fprintf(stderr,"** Warning: failed to parse: \"%s\" as valid minimum value - ignored \n",value);
	option_OK = true;
      }
      
      if (strcmp(option , "MAX") == 0) {
	double max_value;
	if (util_sscanf_double( value , &max_value)) {
	  config->max_value  = max_value;
	  util_bitmask_on( &config->truncation , TRUNCATE_MAX );
	} else
	  fprintf(stderr,"** Warning: failed to parse: \"%s\" as valid maximum value - ignored \n",value);
	option_OK = true;
      }

      if (strcmp(option , "OUTPUT_TRANSFORM") == 0) {
	if (field_trans_table_has_key( field_trans_table , value))
	  config->output_transform = field_trans_table_lookup( field_trans_table , value);
	else {
	  fprintf(stderr,"** Warning: function name:%s not recognized - ignored. \n",value);
	  field_trans_table_fprintf(field_trans_table , stderr);
	}
	option_OK = true;
      }

      if (strcmp(option , "INPUT_TRANSFORM") == 0) {
	if (field_trans_table_has_key( field_trans_table , value))
	  config->input_transform = field_trans_table_lookup( field_trans_table , value);
	else {
	  fprintf(stderr,"** Warning: function name:%s not recognized - ignored. \n",value);
	  field_trans_table_fprintf(field_trans_table , stderr);
	}
	option_OK = true;
      }

      if (strcmp(option , "INIT_TRANSFORM") == 0) {
	if (field_trans_table_has_key( field_trans_table , value))
	  config->init_transform = field_trans_table_lookup( field_trans_table , value);
	else {
	  fprintf(stderr,"** Warning: function name:%s not recognized - ignored. \n",value);
	  field_trans_table_fprintf(field_trans_table , stderr);
	}
	option_OK = true;
      }

      if (strcmp(option , "INIT_FILES") == 0) {
	config->init_file_fmt = path_fmt_alloc_path_fmt( value );
	option_OK = true;
      }

      if (strcmp(option , "MIN_STD") == 0) {
        min_std_file = value;
        option_OK = true;
      }

      if (!option_OK)
	fprintf(stderr,"** Warning: \"%s\" not recognized - ignored \n",option);

      option = hash_iter_get_next_key(iter);
    }
    
    if (min_std_file != NULL) {
      config->min_std = field_alloc( config );
      field_fload(config->min_std , min_std_file );
    }

    hash_iter_free(iter);
    hash_free(opt_hash);
  }
  return config;
}



/*
  Observe that the indices are zero-based, in contrast to those used
  by eclipse which are based on one.

  This function will return an index in the interval: [0...nactive),
  and -1 if i,j,k correspond to an inactive cell.
*/


inline int field_config_active_index(const field_config_type * config , int i , int j , int k) {
  return ecl_grid_get_active_index3( config->grid , i,j,k);
}


/**
    This function checks that i,j,k are in the intervals [0..nx),
    [0..ny) and [0..nz). It does *NOT* check if the corresponding
    index is active.
*/

bool field_config_ijk_valid(const field_config_type * config , int i , int j , int k) {
  return ecl_grid_ijk_valid(config->grid , i,j,k);
}


void field_config_get_ijk( const field_config_type * config , int active_index , int *i , int * j , int * k) {
  ecl_grid_get_ijk1A( config->grid , active_index , i,j,k);
}


field_config_type * field_config_alloc_dynamic(const char * ecl_kw_name , const ecl_grid_type * ecl_grid , field_trans_table_type * trans_table , const stringlist_type * options) {
  field_config_type * config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid , ECL_FILE , ECL_FILE , trans_table , options);
  return config;
}



field_config_type * field_config_alloc_general(const char * ecl_kw_name , 
					       const char * ecl_file    , 
					       const ecl_grid_type * ecl_grid , 
					       ecl_type_enum internal_type , 
					       field_trans_table_type * trans_table , 
					       const stringlist_type * options) {
  field_config_type * config;
  field_file_format_type import_format = UNDEFINED_FORMAT;   /* undefined_format -> automagic guessing of format. */
  field_file_format_type export_format = field_config_default_export_format( ecl_file );

  config = field_config_alloc__(ecl_kw_name , internal_type , ecl_grid , import_format , export_format , trans_table , options);

  return config;
}





/* This interface is just to general */
field_config_type * field_config_alloc_parameter(const char * ecl_kw_name 	      ,
						 const char * ecl_file    	      ,
						 const ecl_grid_type * ecl_grid       ,
						 field_trans_table_type * trans_table ,
						 const stringlist_type * options) {
  field_config_type * config;
  field_file_format_type import_format = UNDEFINED_FORMAT;
  field_file_format_type export_format = field_config_default_export_format( ecl_file );

  config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid , import_format , export_format , trans_table ,options);
  if (config->init_file_fmt == NULL)
    util_abort("%s:(INTERNAL ERROR)  invalid init type \n",__func__);


  return config;
}



bool field_config_write_compressed(const field_config_type * config) { return config->write_compressed; }



void field_config_set_truncation(field_config_type * config , truncation_type truncation, double min_value, double max_value) {
  config->truncation = truncation;
  config->min_value  = min_value;
  config->max_value  = max_value;
}




//void field_config_set_truncation_from_strings(field_config_type * config , const char * _truncation_name , const char ** values) {
//  if (_truncation_name != NULL) {
//    char * truncation_name = util_alloc_strupr_copy( _truncation_name ) ;
//    truncation_type truncation = truncate_none;
//
//    if (strcmp(truncation_name , "NONE") == 0)
//      truncation = truncate_none;
//    else if (strcmp(truncation_name , "MIN") == 0)
//      truncation = truncate_min;
//    else if (strcmp(truncation_name , "MAX") == 0)
//      truncation = truncate_max;
//    else if (strcmp(truncation_name , "MINMAX") == 0)
//      truncation = truncate_minmax;
//    else
//      util_abort("%s: truncation string:%s is not recognized \n",__func__ , _truncation_name);
//
//    {
//      int    value_index = 0;
//      double min_value 	 = -1;
//      double max_value 	 = -1;
//
//      if (truncation & truncate_min) {
//	if (!util_sscanf_double(values[value_index] , &min_value))
//	  util_abort("%s: failed to parse:%s as double \n",__func__ , values[value_index]);
//
//	if (truncation == truncate_minmax)
//	  value_index++;
//      }
//
//      if (truncation & truncate_max) {
//	if (! util_sscanf_double(values[value_index] , &max_value))
//	  util_abort("%s: failed to parse:%s as double \n",__func__ , values[value_index]);
//      }
//
//      field_config_set_truncation(config , truncation , min_value , max_value);
//    }
//    free( truncation_name );
//  }
//}


truncation_type field_config_get_truncation(const field_config_type * config , double * min_value , double * max_value) {
  *min_value = config->min_value;
  *max_value = config->max_value;
  return config->truncation;
}





void field_config_set_io_options(const field_config_type * config , bool *fmt_file ) {
  *fmt_file    = config->fmt_file;
}



void field_config_free(field_config_type * config) {
  util_safe_free(config->ecl_kw_name);
  active_list_free(config->active_list);
  if (config->init_file_fmt != NULL) path_fmt_free( config->init_file_fmt );
  free(config);
}



int field_config_get_volume(const field_config_type * config) {
  return config->nx * config->ny * config->nz;
}



rms_type_enum field_config_get_rms_type(const field_config_type * config) {
  return rms_util_convert_ecl_type(config->internal_ecl_type);
}



ecl_type_enum field_config_get_ecl_type(const field_config_type * config) {
  return config->internal_ecl_type;
}



int field_config_get_byte_size(const field_config_type * config) {
  return config->data_size * config->sizeof_ctype;
}





int field_config_get_sizeof_ctype(const field_config_type * config) { return config->sizeof_ctype; }






/**
   Returns true / false whether a cell is active.
*/
bool field_config_active_cell(const field_config_type * config , int i , int j , int k) {
  int active_index = field_config_active_index(config , i,j,k);
  if (active_index >= 0)
    return true;
  else
    return false;
}




bool field_config_enkf_init(const field_config_type * config) {
  if (config->init_file_fmt != NULL)
    return true;
  else
    return false;
}






char * field_config_alloc_init_file(const field_config_type * config, int iens) {
  return path_fmt_alloc_path(config->init_file_fmt , false , iens);
}






 void field_config_get_dims(const field_config_type * config , int *nx , int *ny , int *nz) {
   *nx = config->nx;
   *ny = config->ny;
   *nz = config->nz;
}







/**
   This function reads a string with i,j,k from the user. All
   characters in the constant sep_set are allowed to separate the
   integers. The function will loop until:

   * Three integers have been succesfully parsed.
   * All numbers are in the (1-nx,1-ny,1-nz) intervals.
   * IFF active_only - only active cells wll be allowed.

   i,j,k and global_index are returned by reference. All pointers can
   be NULL, if you are not interested. An invald global_index is
   returned as -1 (if active_only == false).

   Observe that the user is expected to enter numbers in the interval
   [1..nx],[1..ny],[1..nz], but internaly they are immediately
   converted to zero offset.
*/


void field_config_scanf_ijk(const field_config_type * config , bool active_only , const char * _prompt , int prompt_len , int *_i , int *_j , int *_k , int * _global_index) {
  const char * sep_set = " ,.:";
  char * prompt = util_alloc_sprintf("%s (%d,%d,%d)" , _prompt , config->nx , config->ny , config->nz);
  bool OK;
  int i,j,k,global_index;
  global_index = -1; /* Keep the compiler happy. */

  do {
    char         *input;
    const  char  *current_ptr;
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    input = util_alloc_stdin_line();


    i = -1;
    j = -1;
    k = -1;

    OK = true;
    current_ptr = input;
    current_ptr = util_parse_int(current_ptr , &i , &OK);
    current_ptr = util_skip_sep(current_ptr , sep_set , &OK);
    current_ptr = util_parse_int(current_ptr , &j , &OK);
    current_ptr = util_skip_sep(current_ptr , sep_set , &OK);
    current_ptr = util_parse_int(current_ptr , &k , &OK);
    if (OK)
      if (current_ptr[0] != '\0') OK = false; /* There was something more at the end */

    /* Now we have three valid integers. */

    if (OK) {
      if (i <= 0 || i > config->nx) OK = false;
      if (j <= 0 || j > config->ny) OK = false;
      if (k <= 0 || k > config->nz) OK = false;
      i--; j--; k--;
    }
    /* Now we have three integers in the right interval. */


    if (OK) {
      global_index = field_config_active_index(config , i,j,k);
      if (active_only) {
	if (global_index < 0) {
	  OK = false;
	  printf("Sorry the point: (%d,%d,%d) corresponds to an inactive cell\n" , i + 1 , j+ 1 , k + 1);
	}
      }
    }
    free(input);
  } while (!OK);

  if (_i != NULL) *_i = i;
  if (_j != NULL) *_j = j;
  if (_k != NULL) *_k = k;
  if (_global_index != NULL) *_global_index = global_index;

  free(prompt);
}




/**
   The field_config and field objects are mainly written for use in
   the enkf application. In that setting a field instance is *NOT*
   allowed to write on it's field_config object.

   However, when used in a stand-alone application, i.e. in the
   field_convert program, it is desirable for the field object to be
   allowed to write to / update the field_config object. In an attempt
   to make this reasonably safe you must first call
   field_config_enkf_OFF() to signal that you know what you are doing.

   After you have called field_config_enkf_OFF() you can subsequently
   call field_config_set_key() to change the key of the field_config
   object. This will typically be interesting when an unknown file is
   loaded.

   Currently only the roff loader supports set operations on the
   key. Also it is essential to observe that this will break **HARD**
   is the file contains several parameters - so maybe this whole thing
   is stupid?
*/


void field_config_set_key(field_config_type * config , const char *key) {
  if (config->__enkf_mode)
    util_abort("%s: internal error - must call field_config_enkf_OFF() prior to calling: %s()\n",__func__ , __func__);
  /*
    Should be locked to protect against concurrent access.
  */
  config->ecl_kw_name = util_realloc_string_copy(config->ecl_kw_name , key);
}

const char * field_config_get_key(const field_config_type * field_config) {
  return field_config->ecl_kw_name;
}


void field_config_enkf_OFF(field_config_type * config) {
  if (config->__enkf_mode)
    fprintf(stderr , "** Warning: turning off EnKF mode for field:%s - you better know what you are doing! **\n",config->ecl_kw_name);
  config->__enkf_mode = false;
}


bool field_config_enkf_mode(const field_config_type * config) { return config->__enkf_mode; }


field_func_type * field_config_get_output_transform(const field_config_type * config) {
  return config->output_transform;
}

field_func_type * field_config_get_input_transform(const field_config_type * config) {
  return config->input_transform;
}

field_func_type * field_config_get_init_transform(const field_config_type * config) {
  return config->init_transform;
}

void field_config_set_output_transform(field_config_type * config , field_func_type * func) {
  config->output_transform = func;
}


/*
  This function asserts that a unary function can be applied
  to the field - i.e. that the underlying data_type is ecl_float or ecl_double.
*/
void field_config_assert_unary( const field_config_type * field_config , const char * caller) {
  const ecl_type_enum ecl_type = field_config_get_ecl_type(field_config);
  if (ecl_type == ecl_float_type || ecl_type == ecl_double_type)
    return;
  else
    util_abort("%s: error in:%s unary functions can only be applied on fields of type ecl_float / ecl_double \n",__func__ , caller);
}


/*
   Asserts that two fields can be combined in a binary operation.
*/
void field_config_assert_binary( const field_config_type * config1 , const field_config_type * config2 , const char * caller) {
  field_config_assert_unary(config1 , caller);
  const ecl_type_enum ecl_type1 = config1->internal_ecl_type;
  const ecl_type_enum ecl_type2 = config2->internal_ecl_type;
  const int size1               = config1->data_size;
  const int size2               = config2->data_size;

  if ((ecl_type1 == ecl_type2) && (size1 == size2))
    return;
  else
    util_abort("%s: fields not equal enough - failure in:%s \n",__func__ , caller);
}





//void field_config_activate(field_config_type * config , active_mode_type active_mode , void * active_config) {
//  field_active_type * active = field_active_safe_cast( active_config );
//
//  if (active_mode == ALL_ACTIVE)
//    active_list_set_all_active(config->active_list);
//  else {
//    active_list_reset(config->active_list);
//    if (active_mode == PARTLY_ACTIVE)
//      field_active_update_active_list( active , config->active_list);
//  }
//}



/**
   Parses a string of the type "1,5,6", and returns the indices i,j,k
   by reference. The return value of the function as a whole is
   whether the string constitutes a valid cell:

      0: All is OK.
      1: The string could not pe parsed to three valid integers.
      2: ijk are not in the grid.
      3: ijk correspond to an inactive cell.

   In cases 2 & 3 the i,j,k are valid (in the string-parsing sense).
*/
   


int field_config_parse_user_key(const field_config_type * config, const char * index_key , int *_i , int *_j , int *_k) {
  int      return_value = 0;
  int      length;
  int    * indices = util_sscanf_alloc_active_list(index_key, &length);
  
  if(length != 3)
    return_value = 1;
  else
  {
    
    int i = indices[0] - 1;
    int j = indices[1] - 1;
    int k = indices[2] - 1;
    
    if(field_config_ijk_valid(config, i, j, k)) {
      int active_index = field_config_active_index(config , i,j,k);
      if (active_index < 0)
	return_value = 3;  	/* ijk corresponds to an inactive cell. */
    }  else 
      return_value = 2;         /* ijk is outside the grid. */

    *_i = i;
    *_j = j;
    *_k = k;
  }
  free(indices);
  return return_value;
}



const ecl_grid_type *field_config_get_grid(const field_config_type * config) { return config->grid; }


/*****************************************************************/
SAFE_CAST(field_config , FIELD_CONFIG_ID)
CONFIG_GET_ECL_KW_NAME(field);
GET_DATA_SIZE(field)
VOID_GET_DATA_SIZE(field)
VOID_FREE(field_config)
GET_ACTIVE_LIST(field);

