#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <config.h>
#include <ecl_util.h>
#include <enkf_macros.h>
#include <gen_data_config.h>
#include <enkf_types.h>
#include <pthread.h>
#include <path_fmt.h>
#include <gen_data_active.h>
#include <active_list.h>

#define GEN_DATA_CONFIG_ID 90051
struct gen_data_config_struct {
  CONFIG_STD_FIELDS;
  char                         * key;                   /* The key this gen_data instance is known under - needed for debugging. */
  ecl_type_enum  	         internal_type;         /* The underlying type (float | double) of the data in the corresponding gen_data instances. */
  char           	       * template_buffer;       /* Buffer containing the content of the template - read and internalized at boot time. */
  int            	         template_data_offset;  /* The offset into the template buffer before the data should come. */
  int            	         template_data_skip;    /* The length of data identifier in the template.*/ 
  int            	         template_buffer_size;  /* The total size (bytes) of the template buffer .*/
  path_fmt_type  	       * init_file_fmt;         /* file format for the file used to load the inital values - NULL if the instance is initialized from the forward model. */
  gen_data_file_format_type    	 input_format;          /* The format used for loading gen_data instances when the forward model has completed *AND* for loading the initial files.*/
  gen_data_file_format_type    	 output_format;         /* The format used when gen_data instances are written to disk for the forward model. */
  active_list_type             * active_list;           /* List of (EnKF) active indices. */
  pthread_mutex_t                update_lock;           /* mutex serializing (write) access to the gen_data_config object. */
  int                            __report_step;         /* Internal variable used for run_time checking that all instances have the same size (at the same report_step). */
};

/*****************************************************************/

SAFE_CAST(gen_data_config , GEN_DATA_CONFIG_ID)

gen_data_file_format_type gen_data_config_get_input_format ( const gen_data_config_type * config) { return config->input_format; }
gen_data_file_format_type gen_data_config_get_output_format( const gen_data_config_type * config) { return config->output_format; }

int gen_data_config_get_data_size(const gen_data_config_type * config)   { return config->data_size;     }
int gen_data_config_get_report_step(const gen_data_config_type * config) { return config->__report_step; }


int gen_data_config_get_byte_size(const gen_data_config_type * config) {
  return config->data_size * ecl_util_get_sizeof_ctype(config->internal_type);
}

ecl_type_enum gen_data_config_get_internal_type(const gen_data_config_type * config) {
  return config->internal_type;
}



/**
   Internal consistency checks:

   1. (ecl_file != NULL)                                 => out_format != gen_data_undefined.
   2. ((result_file != NULL) || (init_file_fmt != NULL)) => input_format != gen_data_undefined
   3. (output_format == ASCII_template)                  => (template_ecl_file != NULL) && (template_data_key != NULL)
   4. as_param == true                                   => init_file_fmt != NULL
   5. input_format == ASCII_template                     => INVALID
*/

static gen_data_config_type * gen_data_config_alloc__(const char * key,  
						      bool as_param, 
						      ecl_type_enum internal_type       , 
						      gen_data_file_format_type input_format ,
						      gen_data_file_format_type output_format,
						      const char * init_file_fmt     ,  
						      const char * template_ecl_file , 
						      const char * template_data_key ,
						       const char * ecl_file          ,
						      const char * result_file) {
  
  gen_data_config_type * config = util_malloc(sizeof * config , __func__);
  config->__type_id         = GEN_DATA_CONFIG_ID;
  config->key               = util_alloc_string_copy( key );
  config->data_size  	    = 0;
  config->internal_type     = internal_type;
  config->active_list       = active_list_alloc(0);
  config->input_format      = input_format;
  config->output_format     = output_format;
  config->__report_step     = -1;

  /* Condition 1: */
  if ((ecl_file != NULL) && (output_format == gen_data_undefined))
    util_abort("%s: invalid configuration. When ecl_file != NULL you must explicitly specify an output format with OUTPUT_FORMAT:.\n",__func__);

  /* Condition 2: */
  if (((result_file != NULL) || (init_file_fmt != NULL)) && (input_format == gen_data_undefined))
    util_abort("%s: invalid configuration. When loading with result_file / init_files you must specify an input format with INPUT_FORMAT: \n",__func__);

  /* Condition 3: */
  if (config->output_format == ASCII_template) {
    if (template_ecl_file == NULL)
      util_abort("%s: internal error - when using format ASCII_template you must supply a temlate file \n",__func__);
  } else
    if (template_ecl_file != NULL)
      util_abort("%s: internal error have template and format mismatch \n",__func__);
  
  /* Condition 4: */
  if (as_param)
    if (init_file_fmt == NULL)
      util_abort("%s: when adding a parameter you must supply files to initialize from with INIT_FILES:/path/to/files/with%d \n",__func__);
  
  /* Condition 5: */
  if (input_format == ASCII_template)
    util_abort("%s: Format ASCII_TEMPLATE is not valid as INPUT_FORMAT \n",__func__);
  
  if (template_ecl_file != NULL) {
    char *data_ptr;
    config->template_buffer = util_fread_alloc_file_content( template_ecl_file , NULL , &config->template_buffer_size);
    data_ptr = strstr(config->template_buffer , template_data_key);
    if (data_ptr == NULL) 
      util_abort("%s: template:%s can not be used - could not find data key:%s \n",__func__ , template_ecl_file , template_data_key);
    else {
      config->template_data_offset = data_ptr - config->template_buffer;
      config->template_data_skip   = strlen( template_data_key );
    }
  } else 
    config->template_buffer = NULL;

  if (init_file_fmt != NULL)
    config->init_file_fmt     = path_fmt_alloc_path_fmt( init_file_fmt );
  else
    config->init_file_fmt = NULL;
  
  pthread_mutex_init( &config->update_lock , NULL );

  
  if (config->output_format == gen_data_undefined) 
    config->output_format = config->input_format;
  return config;
}




/**
   This function takes a string representation of one of the
   gen_data_file_format_type values, and returns the corresponding
   integer value.
   
   Will return gen_data_undefined if the string is not recognized,
   calling scope must check on this return value.
*/


static gen_data_file_format_type __gen_data_config_check_format( const char * format ) {
  gen_data_file_format_type type = gen_data_undefined;

  if (strcmp(format , "ASCII") == 0)
    type = ASCII;
  else if (strcmp(format , "ASCII_TEMPLATE") == 0)
    type = ASCII_template;
  else if (strcmp(format , "BINARY_DOUBLE") == 0)
    type = binary_double;
  else if (strcmp(format , "BINARY_FLOAT") == 0)
    type = binary_float;
  
  if (type == gen_data_undefined)
    util_exit("Sorry: format:\"%s\" not recognized - valid values: ASCII / ASCII_TEMPLATE / BINARY_DOUBLE / BINARY_FLOAT \n", format);
  
  return type;
}

/**
   The valid options are:

   INPUT_FORMAT:(ASCII|ASCII_TEMPLATE|BINARY_DOUBLE|BINARY_FLOAT)
   OUTPUT_FORMAT:(ASCII|ASCII_TEMPLATE|BINARY_DOUBLE|BINARY_FLOAT)
   INIT_FILES:/some/path/with/%d
   TEMPLATE:/some/template/file
   KEY:<SomeKeyFoundInTemplate>
   ECL_FILE:<filename to write EnKF ==> Forward model>  (In the case of gen_param - this is extracted in the calling scope).
   RESULT_FILE:<filename to read EnKF <== Forward model> 

*/

gen_data_config_type * gen_data_config_alloc(const char * key , bool as_param , const stringlist_type * options , char **__ecl_file , char ** __result_file) {
  const ecl_type_enum internal_type = ecl_double_type;
  gen_data_config_type * config;
  hash_type * opt_hash = hash_alloc_from_options( options );
  char * result_file   = NULL;
  char * ecl_file      = NULL;

  /* Parsing options */
  {
    gen_data_file_format_type input_format  = gen_data_undefined;
    gen_data_file_format_type output_format = gen_data_undefined;
    char * template_file = NULL;
    char * template_key  = NULL;
    char * init_file_fmt = NULL;

    hash_iter_type * iter = hash_iter_alloc(opt_hash);
    const char * option = hash_iter_get_next_key(iter);
    while (option != NULL) {
      const char * value = hash_get(opt_hash , option);

      /* 
	 That the various options are internally consistent is ensured
	 in the final static allocater.
      */
      if (strcmp(option , "INPUT_FORMAT") == 0) 
	input_format = __gen_data_config_check_format( value );
      
      else if (strcmp(option , "OUTPUT_FORMAT") == 0)
	output_format = __gen_data_config_check_format( value );
      
      else if (strcmp(option , "TEMPLATE") == 0)
	template_file = util_alloc_string_copy( value );
      
      else if (strcmp(option , "KEY") == 0)
	template_key = util_alloc_string_copy( value );
      
      else if (strcmp(option , "INIT_FILES") == 0)
	init_file_fmt = util_alloc_string_copy( value );

      else if ((__ecl_file != NULL) && ((strcmp(option , "ECL_FILE") == 0)))
	ecl_file = util_alloc_string_copy( value );
      
      else if ((__result_file != NULL) && ((strcmp(option , "RESULT_FILE") == 0)))
	result_file = util_alloc_string_copy( value );
      
      else
	fprintf(stderr , "%s: Warning: \'%s:%s\' not recognized as valid option - ignored \n",__func__ , option , value);

      
      option = hash_iter_get_next_key(iter);
    } 
    config = gen_data_config_alloc__(key , as_param , internal_type , input_format , output_format , init_file_fmt , template_file , template_key , ecl_file , result_file);
    util_safe_free( init_file_fmt );
    util_safe_free( template_file );
    util_safe_free( template_key );
    hash_iter_free( iter );
  }
  hash_free(opt_hash);

  /* These must be returned to the enkf_node layer - UGGGLY */
  if (__ecl_file    != NULL) *__ecl_file    = ecl_file;
  if (__result_file != NULL) *__result_file = result_file;
  return config;
}




void gen_data_config_free(gen_data_config_type * config) {
  active_list_free(config->active_list);
  if (config->init_file_fmt != NULL) path_fmt_free(config->init_file_fmt);
  util_safe_free( config->key );
  free(config);
}




/**
   This function gets a size (from a gen_data) instance, and verifies
   that the size agrees with the currently stored size and
   report_step. If the report_step is new we just recordthe new info,
   otherwise it will break hard.
*/


/**
   Does not work properly with:
   
   1. keep_run_path - the load_file will be left hanging around - and loaded again and again.
   2. Doing forward several steps - how to (time)index the files?
     
*/

/* Locking is completelt broken here ... */
void gen_data_config_assert_size(gen_data_config_type * config , int data_size, int report_step) {
  pthread_mutex_lock( &config->update_lock );
  {

    if (report_step != config->__report_step) {
      config->data_size     = data_size; 
      config->__report_step = report_step;
      active_list_set_data_size( config->active_list , data_size );
    } else if (config->data_size != data_size) {
      util_abort("%s: Size mismatch when loading from file - got %d elements - expected:%d [report_step:%d] \n",
		 __func__ , 
		 data_size , 
		 config->data_size, 
		 report_step);
    }

  }
  pthread_mutex_unlock( &config->update_lock );
}





char * gen_data_config_alloc_initfile(const gen_data_config_type * config , int iens) {
  if (config->init_file_fmt != NULL) {
    char * initfile = path_fmt_alloc_path(config->init_file_fmt , false , iens);
    return initfile;
  } else 
    return NULL; 
}




void gen_data_config_get_template_data( const gen_data_config_type * config , 
					char ** template_buffer    , 
					int * template_data_offset , 
					int * template_buffer_size , 
					int * template_data_skip) {
  
  *template_buffer      = config->template_buffer;
  *template_data_offset = config->template_data_offset;
  *template_buffer_size = config->template_buffer_size;
  *template_data_skip   = config->template_data_skip;
  
}




void gen_data_config_activate(gen_data_config_type * config , active_mode_type active_mode , void * active_config) {
  gen_data_active_type * active = gen_data_active_safe_cast( active_config );

  if (active_mode == ALL_ACTIVE)
    active_list_set_all_active(config->active_list);
  else {
    active_list_reset(config->active_list);
    if (active_mode == PARTLY_ACTIVE) 
      gen_data_active_update_active_list( active , config->active_list);
  }
    
}


const char * gen_data_config_get_key( const gen_data_config_type * config) {
  return config->key;
}


/*****************************************************************/

VOID_FREE(gen_data_config)
GET_ACTIVE_LIST(gen_data)
VOID_CONFIG_ACTIVATE(gen_data)
