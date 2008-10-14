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

struct field_config_struct {
  CONFIG_STD_FIELDS;
  int nx,ny,nz;                       /* The number of elements in the three directions. */ 
  int sx,sy,sz;                       /* The stride in the various directions, i.e. when adressed as one long vector in memory you jump sz elements to iterate along the z direction. */ 
  const int *index_map;

  bool        * enkf_active;          /* Whether a certain cell is active or not - EnKF wise.*/
  bool          enkf_all_active;      /* Performance gain when all cells are active. */
  void 	      * min_value;
  void        * max_value;
  int           sizeof_ctype;

  field_file_format_type  export_format;
  field_file_format_type  import_format;    
  ecl_type_enum           internal_ecl_type;
  ecl_type_enum           export_ecl_type;
  field_init_type         init_type; 
  char          	* base_file;
  char          	* perturbation_config_file;
  char                  * layer_config_file;  
  path_fmt_type         * init_file_fmt;

  bool __enkf_mode;  /* See doc of functions field_config_set_key() / field_config_enkf_OFF() */
  bool fmt_file;
  bool endian_swap;
  bool limits_set;
  bool write_compressed;
  bool add_perturbation;

  field_func_type         * output_transform;     /* Function to apply to the data before they are exported - NULL: no transform. */
};



/*****************************************************************/


void field_config_set_ecl_kw_name(field_config_type * config , const char * ecl_kw_name) {
  config->ecl_kw_name = util_realloc_string_copy(config->ecl_kw_name , ecl_kw_name);
}


static void field_config_assert_ijk(const field_config_type * config , int i , int j , int k) {
  if (i < 0 || i >= config->nx) util_abort("%s: i:%d outside valid range: [0,%d) \n",__func__ , i , config->nx);
  if (j < 0 || j >= config->ny) util_abort("%s: j:%d outside valid range: [0,%d) \n",__func__ , j , config->ny);
  if (k < 0 || k >= config->nz) util_abort("%s: k:%d outside valid range: [0,%d) \n",__func__ , k , config->nz);
}


void field_config_set_ecl_type(field_config_type * config , ecl_type_enum ecl_type) {
  config->internal_ecl_type     = ecl_type;
  config->sizeof_ctype = ecl_util_get_sizeof_ctype(ecl_type);
}



static const char * field_config_file_type_string(field_file_format_type file_type) {
  switch (file_type) {
  case(rms_roff_file):
    return "Binary ROFF file from RMS";
    break;
  case(ecl_kw_file):
    return "ECLIPSE file in restart format";
    break;
  case(ecl_kw_file_all_cells):
    return "ECLIPSE file in restart format (all cells)";
    break;
  case(ecl_kw_file_active_cells):
    return "ECLIPSE file in restart format (active cells)";
    break;
  case(ecl_grdecl_file):
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
  if (file_type == rms_roff_file) {
    if (upper_case)
      return "ROFF";
    else
      return "roff";
  } else if (file_type == ecl_grdecl_file) {
    if (upper_case)
      return "GRDECL"; 
    else
      return "grdecl";
  } else
    return NULL;
}




static bool field_config_valid_file_type(field_file_format_type file_type, bool import) {
  if (import) {
    if (file_type == rms_roff_file || file_type == ecl_kw_file || file_type == ecl_grdecl_file)
      return true;
    else
      return false;
  } else {
    if (file_type == rms_roff_file || file_type == ecl_kw_file_active_cells || file_type == ecl_kw_file_all_cells || file_type == ecl_grdecl_file)
      return true;
    else
      return false;
  }
}




/**
   This function prompts the user for a file type. 

   If the parameter 'import' is true we provide the alternative
   ecl_kw_file (in that case the program itself will determine
   whether) the file contains all cells (i.e. PERMX) or only active
   cells (i.e. pressure).

   If the parameter 'import' is true the user must specify whether we
   are considering all cells, or only active cells.
*/

field_file_format_type field_config_manual_file_type(const char * prompt , bool import) {
  int int_file_type;
  printf("\n%s\n",prompt);
  printf("----------------------------------------------------------------\n");
  printf(" %3d: %s.\n" , rms_roff_file   , field_config_file_type_string(rms_roff_file));
  if (import)
    printf(" %3d: %s.\n" , ecl_kw_file     , field_config_file_type_string(ecl_kw_file));
  else {
    printf(" %3d: %s.\n" , ecl_kw_file_active_cells  , field_config_file_type_string(ecl_kw_file_active_cells));      
    printf(" %3d: %s.\n" , ecl_kw_file_all_cells     , field_config_file_type_string(ecl_kw_file_all_cells));
  }
  printf(" %3d: %s.\n" , ecl_grdecl_file , field_config_file_type_string(ecl_grdecl_file));
  printf("----------------------------------------------------------------\n");
  do {
    int_file_type = util_scanf_int("" , 2);
    if (!field_config_valid_file_type(int_file_type, import))
      int_file_type = undefined_format;
  } while(int_file_type == undefined_format);
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
field_file_format_type field_config_guess_file_type(const char * filename , bool endian_flip) {
  bool fmt_file = util_fmt_bit8(filename );
  FILE * stream = util_fopen(filename , "r");

  field_file_format_type file_type;
  if (ecl_kw_is_kw_file(stream , fmt_file , endian_flip))
    file_type = ecl_kw_file;
  else if (rms_file_is_roff(stream))
    file_type = rms_roff_file;
  else if (ecl_kw_is_grdecl_file(stream))  /* This is the weakest test - and should be last in a cascading if / else hierarchy. */
    file_type = ecl_grdecl_file;
  else 
    file_type = undefined_format;              /* MUST Check on this return value */

  fclose(stream);
  return file_type;
}



field_file_format_type field_config_get_export_format(const field_config_type * field_config) {
  return field_config->export_format;
}

field_file_format_type field_config_get_import_format(const field_config_type * field_config) {
  return field_config->import_format;
}



static field_config_type * field_config_alloc__(const char * ecl_kw_name , ecl_type_enum ecl_type , const ecl_grid_type * ecl_grid , field_file_format_type import_format , field_file_format_type export_format) {
  field_config_type *config = util_malloc(sizeof *config, __func__);
  
  /*
    Observe that size is the number of *ACTIVCE* cells,
    and generally *not* equal to nx*ny*nz.
  */
  config->export_format = export_format;
  config->import_format = import_format; 
  
  config->base_file                = NULL;
  config->perturbation_config_file = NULL;
  config->layer_config_file        = NULL;
  
  config->ecl_kw_name = NULL;
  field_config_set_ecl_kw_name(config , ecl_kw_name);
  field_config_set_ecl_type(config , ecl_type);

  ecl_grid_get_dims(ecl_grid , &config->nx , &config->ny , &config->nz , &config->data_size);
  config->index_map = ecl_grid_get_index_map_ref(ecl_grid);
  
  config->sx = 1;
  config->sy = config->nx;
  config->sz = config->nx * config->ny;

  config->__enkf_mode              = true;
  config->fmt_file    	      	   = false;
  config->endian_swap 	      	   = true;
  config->limits_set  	      	   = false;
  config->min_value   	      	   = util_malloc(config->sizeof_ctype , __func__);
  config->max_value   	      	   = util_malloc(config->sizeof_ctype , __func__);
  config->write_compressed    	   = true;
  config->base_file                = NULL;
  config->perturbation_config_file = NULL;
  config->layer_config_file        = NULL;
  config->init_file_fmt            = NULL;
  config->enkf_active              = util_malloc(config->data_size * sizeof * config->enkf_active , __func__);
  config->output_transform         = NULL;
  field_config_set_all_active(config);
  return config;
}


field_config_type * field_config_alloc_complete(const char * ecl_kw_name , ecl_type_enum ecl_type , const ecl_grid_type * ecl_grid , field_file_format_type import_format , field_file_format_type export_format) {
  return field_config_alloc__(ecl_kw_name , ecl_type , ecl_grid , import_format , export_format);
}


static void field_config_set_all_active__(field_config_type * field_config, bool active) {
  int i; 
  for (i = 0; i < field_config->data_size; i++)
    field_config->enkf_active[i] = active;
  field_config->enkf_all_active = active;
}



void field_config_set_all_active(field_config_type * field) {
  field_config_set_all_active__(field , true);
}

/*
  Observe that the indices are zero-based, in contrast to those used
  by eclipse which are based on one.
*/
inline int field_config_global_index(const field_config_type * config , int i , int j , int k) {
  field_config_assert_ijk(config , i , j , k);
  return config->index_map[ k * config->nx * config->ny + j * config->nx + i];
}



/**
   This function sets the config->enkf_active pointer. The indicies mentioned in
   active_index_list are set to true, the remaining are set to false.

   Observe that the indices i,j and k are __zero__ based.
*/

void field_config_set_iactive(field_config_type * config , int num_active , const int * i , const int *j , const int *k) {
  int index;
  field_config_set_all_active__(config , false);
  for (index = 0; index < num_active; index++) {
    const int global_index = field_config_global_index(config , i[index] , j[index] , k[index]);
    if (global_index < 0)
      fprintf(stderr,"** Warning cell: (%d,%d,%d) is inactive\n",i[index] , j[index] , k[index]);
    else 
      config->enkf_active[global_index]= true;
  }
}


/**
   If ALL cells are active this function returns NULL, which again
   means that a faster serialize/deserialize routine can be used. If
   _not_ all cells are active the function will return the enkf_active vector.
*/

const bool * field_config_get_iactive(const field_config_type * config) {
  if (config->enkf_all_active)
    return NULL; /* Allows for a short-circuit in serialize/deserialize */
  else
    return config->enkf_active;
}



field_config_type * field_config_alloc_dynamic(const char * ecl_kw_name , const ecl_grid_type * ecl_grid) {
  field_config_type * config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid , ecl_restart_block , undefined_format);
  config->init_type         = none;
  config->export_format     = ecl_kw_file_active_cells;
  return config;
}



field_config_type * field_config_alloc_general(const char * ecl_kw_name , const ecl_grid_type * ecl_grid , const char * init_fmt) {
  field_config_type * config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid , ecl_restart_block , undefined_format);
  config->init_type         = load_unique;
  /*****************************************************************/
  /* Hardcoded modell error requirements */
  config->export_format     = ecl_kw_file_active_cells;
  config->import_format     = ecl_kw_file;
  /*****************************************************************/

  config->init_file_fmt     = path_fmt_alloc_path_fmt( init_fmt );
  return config;
}



field_config_type * field_config_alloc_parameter_no_init(const char * ecl_kw_name, const ecl_grid_type * ecl_grid , ecl_type_enum internal_type) {
  field_config_type * config = field_config_alloc__(ecl_kw_name , internal_type , ecl_grid , undefined_format , undefined_format);
  config->init_type          = none;
  return config;
}



/* This interface is just to general */
#define ASSERT_CONFIG_FILE(index , len) if (index >= len) { fprintf(stderr,"%s: lacking configuration information - aborting \n",__func__); abort(); }
field_config_type * field_config_alloc_parameter(const char * ecl_kw_name , const char * ecl_file , const char * output_transform_name, 
						 const ecl_grid_type * ecl_grid ,field_init_type init_type , int config_len , const char ** config_files) {
  field_config_type * config;
  field_file_format_type export_format;
  {
    char * extension;
    util_alloc_file_components(ecl_file , NULL , NULL , &extension);
    util_strupr( extension );
    if (strcmp(extension , "GRDECL") == 0)
      export_format = ecl_grdecl_file;
    else
      export_format = ecl_kw_file_all_cells;
    free(extension);
  }

  config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid , undefined_format , export_format);
  config->init_type = init_type;
  if (init_type == none) {
    fprintf(stderr,"%s: invalid init type \n",__func__);
    abort();
  }
  {
    int config_index = 0;
    if (init_type & load_unique) {
      ASSERT_CONFIG_FILE(config_index , config_len);
      config->init_file_fmt = path_fmt_alloc_path_fmt(config_files[config_index]);
      config_index++;
    }

    if (init_type & load_base_case) {
      ASSERT_CONFIG_FILE(config_index , config_len);
      config->base_file = util_alloc_string_copy(config_files[config_index]);
      config_index++;
    }

    if (init_type & layer_trends) {
      ASSERT_CONFIG_FILE(config_index , config_len);
      config->layer_config_file = util_alloc_string_copy(config_files[config_index]);
      config_index++;
    }

    if (init_type & gaussian_perturbations) {
      ASSERT_CONFIG_FILE(config_index , config_len);
      config->perturbation_config_file = util_alloc_string_copy(config_files[config_index]);
      config_index++;
    }
  }
  {
    char * trans = util_alloc_strupr_copy( output_transform_name );
    field_func_type * func = NULL;
    if (strcmp(trans , "NULL") == 0)
      func = NULL;
    else if (strcmp(trans , "NONE") == 0)
      func = NULL;
    else if (strcmp(trans , "EXP") == 0)
      func = expf;  /* The most common internal implementation is probably float. */
    else 
      util_exit("%s: sorry - function_name:%s not recognized \n",__func__ , output_transform_name); 
    field_config_set_output_transform( config , func );
    free(trans);
  }

  return config;
}
#undef ASSERT_CONFIG_FILE



bool field_config_write_compressed(const field_config_type * config) { return config->write_compressed; }



void field_config_set_limits(field_config_type * config , void * min_value , void * max_value) {
  memcpy(config->min_value , min_value , config->sizeof_ctype);
  memcpy(config->max_value , max_value , config->sizeof_ctype);
  config->limits_set = true;
}



void field_config_apply_limits(const field_config_type * config, void * _data) {
  if (config->internal_ecl_type != ecl_double_type) {
    fprintf(stderr,"%s: sorry - limits only implemented for double fields currently \n",__func__);
    abort();
  }

  if (config->limits_set) {
    switch (config->internal_ecl_type) {
    case(ecl_double_type):
      {
	double *data = (double *) _data;
	int i;
	for (i = 0; i < config->data_size; i++) 
	  util_apply_double_limits(&data[i] , *((double *) config->min_value) , *((double *) config->max_value));
      }
      break;
    case(ecl_float_type):
      {
	float *data = (float *) _data;
	int i;
	for (i = 0; i < config->data_size; i++) 
	  util_apply_float_limits(&data[i] , *((float *) config->min_value) , *((float *) config->max_value));
      }
      break;
    case(ecl_int_type):
      {
	int *data = (int *) _data;
	int i;
	for (i = 0; i < config->data_size; i++) 
	  util_apply_int_limits(&data[i] , *((int *) config->min_value) , *((int *) config->max_value));
      }
      break;
    default:
      fprintf(stderr,"%s field limits only applied for int/double/float fields - aborting \n",__func__);
      abort();
    }
  } else {
    fprintf(stderr,"%s: must set limits with a call to : field_config_set_limits() first - aborting \n",__func__);
    abort();
  }
}



void field_config_set_io_options(const field_config_type * config , bool *fmt_file , bool * endian_swap) {
  *fmt_file    = config->fmt_file;
  *endian_swap = config->endian_swap;
  /*
   *ecl_type    = config->internal_ecl_type;
   */
}



void field_config_free(field_config_type * config) {
  free(config->min_value);
  free(config->max_value);
  util_safe_free(config->ecl_kw_name);
  util_safe_free(config->base_file);
  util_safe_free(config->perturbation_config_file);
  util_safe_free(config->layer_config_file);
  if (config->init_file_fmt != NULL) path_fmt_free( config->init_file_fmt );
  free(config->enkf_active);
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



int field_config_get_active_size(const field_config_type * config) {
  return config->data_size;
}



int field_config_get_sizeof_ctype(const field_config_type * config) { return config->sizeof_ctype; }






/**
   Returns true / false whether a cell is active. 
*/
bool field_config_active_cell(const field_config_type * config , int i , int j , int k) {
  int global_index = field_config_global_index(config , i,j,k);
  if (global_index >= 0)
    return true;
  else
    return false;
}




field_init_type field_config_get_init_type(const field_config_type * config) {
  return config->init_type;
}



bool field_config_get_endian_swap(const field_config_type * config) {
  return config->endian_swap;
}



char * field_config_alloc_init_file(const field_config_type * config, int iens) {
  return path_fmt_alloc_path(config->init_file_fmt , false , iens);
}



/*
  TODO

  This needs to be rewamped or renamed.
  
  The name "get" is misleading in a function that uses a linear lookup in a potentially very large table.

  The "fix" is probably to have an index_map_inv in the ecl_grid type which is alloc'ed simultaneously
  as the the index map. A global/active_index can then be translated into a "natural ordering" index, from
  which an ijk triplet quickly can be computed.
*/
void field_config_get_ijk(const field_config_type * config , int global_index, int *_i , int *_j , int *_k) {
  int i,j,k;
  if (global_index >= config->data_size || global_index < 0) {
    fprintf(stderr,"%s: global_index: %d is not in intervale [0,%d) - aborting \n",__func__ , global_index , config->data_size);
    abort();
  }
  
  for (k=0; k < config->nz; k++)
    for (j=0; j < config->ny; j++)
      for (i=0; i < config->nx; i++)
	if (field_config_global_index(config , i,j,k) == global_index) {
	  *_i = i;
	  *_j = j;
	  *_k = k;
	  return;
	}
  util_abort("%s: should not have arrived here - something wrong with the index_map?? \n",__func__);
}



 void field_config_get_dims(const field_config_type * config , int *nx , int *ny , int *nz) {
   *nx = config->nx;
   *ny = config->ny;
   *nz = config->nz;
}


static const char * __parse_number(const char * s , int * value, bool *OK) {
  if (*OK) {
    char * error_ptr;
    *value = strtol(s , &error_ptr , 10);
    if (error_ptr == s) *OK = false;
    return error_ptr;
  } else
    return NULL;
}



static const char * __skip_sep(const char * s, const char * sep_set, bool *OK) {
  if (*OK) {
    int sep_length = strspn(s , sep_set);
    if (sep_length == 0)
      *OK = false;
    return &s[sep_length];
  } else 
    return NULL;
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
    current_ptr = __parse_number(current_ptr , &i , &OK);  
    current_ptr = __skip_sep(current_ptr , sep_set , &OK); 
    current_ptr = __parse_number(current_ptr , &j , &OK);  
    current_ptr = __skip_sep(current_ptr , sep_set , &OK); 
    current_ptr = __parse_number(current_ptr , &k , &OK);  
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
      global_index = field_config_global_index(config , i,j,k);
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




/*****************************************************************/
CONFIG_GET_ECL_KW_NAME(field);
GET_DATA_SIZE(field)
VOID_FREE(field_config)
