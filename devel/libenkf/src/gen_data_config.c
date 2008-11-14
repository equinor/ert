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
  ecl_type_enum  	  internal_type;         /* The underlying type (float | double) of the data in the corresponding gen_data instances. */
  char           	* template_buffer;   	 /* Buffer containing the content of the template - read and internalized at boot time. */
  int            	  template_data_offset;  /* The offset into to the template buffer before the data should come. */
  int            	  template_data_skip;    /* The length of data identifier in the template.*/ 
  int            	  template_buffer_size;  /* The total size (bytes) of the template buffer .*/
  path_fmt_type  	* init_file_fmt;         /* file format for the file used to load the inital values. */
  gen_data_format_type    input_format;          /* The format used for loading gen_data instances when the forward model has completed *AND* for loading the initial files.*/
  gen_data_format_type    output_format;         /* The format used when gen_data instances are written to disk for the forward model. */

  active_list_type      * active_list;           /* List of (EnKF) active indices. */
  pthread_mutex_t  update_lock;
};

/*****************************************************************/

SAFE_CAST(gen_data_config , GEN_DATA_CONFIG_ID)

gen_data_format_type gen_data_config_get_input_format ( const gen_data_config_type * config) { return config->input_format; }
gen_data_format_type gen_data_config_get_output_format( const gen_data_config_type * config) { return config->output_format; }

int gen_data_config_get_data_size(const gen_data_config_type * config) { return config->data_size; }


int gen_data_config_get_byte_size(const gen_data_config_type * config) {
  return config->data_size * ecl_util_get_sizeof_ctype(config->internal_type);
}

ecl_type_enum gen_data_config_get_internal_type(const gen_data_config_type * config) {
  return config->internal_type;
}


static gen_data_config_type * gen_data_config_alloc__( ecl_type_enum internal_type       , 
						       gen_data_format_type input_format ,
						       gen_data_format_type output_format,
						       const char * init_file_fmt     ,  
						       const char * template_ecl_file , 
						       const char * template_data_key ) {

  gen_data_config_type * config = util_malloc(sizeof * config , __func__);
  config->__type_id         = GEN_DATA_CONFIG_ID;
  config->data_size  	    = 0;
  config->internal_type     = internal_type;
  config->active_list       = active_list_alloc(0);
  config->input_format      = input_format;
  config->output_format     = output_format;

  if (config->output_format == ASCII_template) {
    if (template_ecl_file == NULL)
      util_abort("%s: internal error - when using format ASCII_template you must supply a temlate file \n",__func__);
  } else
    if (template_ecl_file != NULL)
      util_abort("%s: internal error have template and format mismatch \n",__func__);
  

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
  return config;
}


/**
   Observe that the two allocators xxx_alloc() and
   xxx_alloc_with_template() do *NOT* have to know whether the object
   should subsequently be used as a parameter or as dynamic data.
*/

gen_data_config_type * gen_data_config_alloc(gen_data_format_type input_format ,   
					     gen_data_format_type output_format,
					     const char * init_file_fmt) {
  return gen_data_config_alloc__(ecl_double_type , input_format , output_format , init_file_fmt , NULL , NULL);
}



gen_data_config_type * gen_data_config_alloc_with_template(gen_data_format_type input_format,
							   const char * template_file     , 
							   const char * template_data_key , 
							   const char * init_file_fmt) {
  return gen_data_config_alloc__(ecl_double_type , input_format , ASCII_template , init_file_fmt , template_file , template_data_key);
}


void gen_data_config_free(gen_data_config_type * config) {
  active_list_free(config->active_list);
  if (config->init_file_fmt != NULL) path_fmt_free(config->init_file_fmt);
  free(config);
}


gen_data_config_type * gen_data_config_fscanf_alloc(const char * config_file) {
  return NULL;
}



/**
   This function gets a size (from a gen_data) instance, and verifies
   that the size agrees with the currently stored size - if not it
   will break HARD.
*/


void gen_data_config_assert_size(gen_data_config_type * config , int size) {
  pthread_mutex_lock( &config->update_lock );
  {
    if (config->data_size == 0) /* 0 means not yet initialized */
      config->data_size = size; 
    else 
      if (config->data_size != size)
	util_abort("%s: Size mismatch when loading from file got %d elements - expected:%d \n",__func__ , size , config->data_size);
    active_list_set_data_size( config->active_list , size );
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

  if (active_mode == all_active)
    active_list_set_all_active(config->active_list);
  else {
    active_list_reset(config->active_list);
    if (active_mode == partly_active) 
      gen_data_active_update_active_list( active , config->active_list);
  }
    
}



/*****************************************************************/

VOID_FREE(gen_data_config)
GET_ACTIVE_LIST(gen_data)
VOID_CONFIG_ACTIVATE(gen_data)
