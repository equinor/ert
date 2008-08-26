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
  config->ecl_type     = ecl_type;
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
  printf("%s\n",prompt);
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
      int_file_type = unknown_file;
  } while(int_file_type == unknown_file);
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
    file_type = unknown_file;              /* MUST Check on this return value */

  fclose(stream);
  return file_type;
}



field_file_format_type field_config_get_ecl_export_format(const field_config_type * field_config) {
  return field_config->ecl_export_format;
}



static field_config_type * field_config_alloc__(const char * ecl_kw_name , ecl_type_enum ecl_type , const ecl_grid_type * ecl_grid) {
  field_config_type *config = util_malloc(sizeof *config, __func__);
  
  /*
    Observe that size is the number of *ACTIVCE* cells,
    and generally *not* equal to nx*ny*nz.
  */
  config->ecl_export_format        = ecl_kw_file_all_cells; 
  /*
  config->ecl_export_format        = ecl_grdecl_format; 
  */
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
  field_config_set_all_active(config);
  return config;
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
   active_index_list are set to true, the remaining is set to false.

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
  field_config_type * config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid);
  config->logmode           = 0;
  config->init_type         = none;
  config->ecl_export_format = ecl_kw_file_active_cells;
  return config;
}



field_config_type * field_config_alloc_parameter_no_init(const char * ecl_kw_name, const ecl_grid_type * ecl_grid) {
  field_config_type * config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid);
  config->logmode            = 0;
  config->init_type          = none;
  return config;
}



#define ASSERT_CONFIG_FILE(index , len) if (index >= len) { fprintf(stderr,"%s: lacking configuration information - aborting \n",__func__); abort(); }
field_config_type * field_config_alloc_parameter(const char * ecl_kw_name , const ecl_grid_type * ecl_grid , int logmode , field_init_type init_type , int config_len , const char ** config_files) {
  field_config_type * config = field_config_alloc__(ecl_kw_name , ecl_float_type , ecl_grid);
  config->logmode   = logmode;
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
  if (config->ecl_type != ecl_double_type) {
    fprintf(stderr,"%s: sorry - limits only implemented for double fields currently \n",__func__);
    abort();
  }

  if (config->limits_set) {
    switch (config->ecl_type) {
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
   *ecl_type    = config->ecl_type;
   */
}



void field_config_free(field_config_type * config) {
  free(config->min_value);
  free(config->max_value);
  util_safe_free(config->ecl_kw_name);
  util_safe_free(config->base_file);
  util_safe_free(config->perturbation_config_file);
  util_safe_free(config->layer_config_file);
  free(config->enkf_active);
  free(config);
}
  


int field_config_get_volume(const field_config_type * config) {
  return config->nx * config->ny * config->nz;
}



rms_type_enum field_config_get_rms_type(const field_config_type * config) {
  return rms_util_convert_ecl_type(config->ecl_type);
}



ecl_type_enum field_config_get_ecl_type(const field_config_type * config) {
  return config->ecl_type;
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


void field_config_enkf_OFF(field_config_type * config) {
  if (config->__enkf_mode)
    fprintf(stderr , "** Warning: turning off EnKF mode for field:%s - you better know what you are doing! **\n",config->ecl_kw_name);
  config->__enkf_mode = false;
}


bool field_config_enkf_mode(const field_config_type * config) { return config->__enkf_mode; }


/*****************************************************************/
CONFIG_GET_ECL_KW_NAME(field);
GET_DATA_SIZE(field)
VOID_FREE(field_config)
