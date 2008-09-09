#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <config.h>
#include <ecl_util.h>
#include <enkf_macros.h>
#include <gen_param_config.h>
#include <enkf_types.h>
#include <pthread.h>
#include <path_fmt.h>



struct gen_param_config_struct {
  CONFIG_STD_FIELDS;
  ecl_type_enum    ecl_type;          	  /* The underlying type (float | double) of the data in the corresponding gen_param instances. */
  bool    	 * iactive;           	  /* MAP of active / inactive observations EnKF wise - for all active it can(should) be NULL.*/
  char           * template_buffer;   	  /* Buffer containing the content of the template - read and internalized at boot time. */
  int              template_data_offset;  /* The offset into to the template buffer before the data should come. */
  int              template_data_skip;    /* The length of data identifier in the template.*/ 
  int              template_buffer_size;  /* The total size (bytes) of the template buffer .*/
  path_fmt_type  * init_file_fmt;         /* file format for the file used to load the inital parameter value. */
  pthread_mutex_t  update_lock;
};



int gen_param_config_get_data_size(const gen_param_config_type * config) { return config->data_size; }


int gen_param_config_get_byte_size(const gen_param_config_type * config) {
  return config->data_size * ecl_util_get_sizeof_ctype(config->ecl_type);
}

ecl_type_enum gen_param_config_get_ecl_type(const gen_param_config_type * config) {
  return config->ecl_type;
}


static gen_param_config_type * gen_param_config_alloc__( ecl_type_enum ecl_type , enkf_var_type var_type , const char * init_file_fmt , const char * template_ecl_file , const char * template_data_key) {
  gen_param_config_type * config = util_malloc(sizeof * config , __func__);
  config->data_size  	    = 0;
  config->var_type   	    = var_type;
  config->ecl_type          = ecl_type;
  config->iactive           = NULL;
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
    
  config->init_file_fmt     = path_fmt_alloc_path_fmt( init_file_fmt );

  pthread_mutex_init( &config->update_lock , NULL );
  return config;
}


gen_param_config_type * gen_param_config_alloc(const char * init_file_fmt , const char * template_ecl_file) {
  return gen_param_config_alloc__(ecl_double_type , parameter , init_file_fmt , template_ecl_file , "<DATA>" );
}



void gen_param_config_free(gen_param_config_type * config) {
  util_safe_free(config->iactive);
  util_safe_free(config->template_buffer);
  path_fmt_free(config->init_file_fmt);
  free(config);
}




/**
   This function gets a size (from a gen_param) instance, and verifies
   that the size agrees with the currently stored size - if not it
   will break HARD.
*/


void gen_param_config_assert_size(gen_param_config_type * config , int size, const char * init_file) {
  pthread_mutex_lock( &config->update_lock );
  {
    if (config->data_size == 0) /* 0 means not yet initialized */
      config->data_size = size; 
    else 
      if (config->data_size != size)
	util_abort("%s: Size mismatch when loading from:%s got %d elements - expected:%d \n",__func__ , init_file , size , config->data_size);
  }
  pthread_mutex_unlock( &config->update_lock );
}



const bool * gen_param_config_get_iactive(const gen_param_config_type * config) { return config->iactive; }

char * gen_param_config_alloc_initfile(const gen_param_config_type * config , int iens) {
  char * initfile = path_fmt_alloc_path(config->init_file_fmt , false , iens);
  return initfile;
}



/** 
    This function is used to write a gen_param instance to disk, for
    future consumption by the forward model. This function will be
    called by a gen_param instance, and the data will come from that instance.
    
    The data written by this function will be a ascii vector of
    floating point numbers (formatted with %g), each number on a
    separate line. In addition you can have a template file:

    template
    --------
    Header1
    Header2
    xxxx
    HeaderN
    <DATA>
    Tail
    --------
    
    Then the data vector will be inserted at the location of the
    <DATA> string. If more advanced manipulation of the output is
    required you must install a job in the forward model to handle it.
*/


void gen_param_config_ecl_write(const gen_param_config_type * config , const char * eclfile , char * data) {
  FILE * stream   = util_fopen(eclfile , "w");
  if (config->template_buffer != NULL)
    util_fwrite( config->template_buffer , 1 , config->template_data_offset , stream , __func__);

  {
    int i;
    if (config->ecl_type == ecl_float_type) {
      float * float_data = (float *) data;
      for (i=0; i < config->data_size; i++)
	fprintf(stream , "%g\n",float_data[i]);
    } else if (config->ecl_type == ecl_double_type) {
      double * double_data = (double *) data;
      for (i=0; i < config->data_size; i++)
	fprintf(stream , "%g\n",double_data[i]);
    } else 
      util_abort("%s: internal error - wrong type \n",__func__);
  }

  if (config->template_buffer != NULL) {
    int new_offset = config->template_data_offset + config->template_data_skip;
    util_fwrite( &config->template_buffer[new_offset] , 1 , config->template_buffer_size - new_offset , stream , __func__);
  }
  fclose(stream);
}


VOID_FREE(gen_param_config)
     
