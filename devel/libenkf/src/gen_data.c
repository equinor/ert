#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <ecl_sum.h>
#include <fortio.h>
#include <ecl_util.h>

#include <enkf_types.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <gen_data_config.h>
#include <gen_data.h>
#include <enkf_serialize.h>



/**
   This file implements a "general parameter" type. The use of
   gen_data instances is roughly as follows:

   1. Some external program creates realizations of the parameter.

   2. The EnKF program loads the parameters in the initialization
      step; EnKF has by construction __NO_WAY__ to sample a gen_data
      instance.

   3. The parameter is updated by EnKF in the normal way, and then
      written out to the run_path directories. If you need fancy
      post-processing of the parameter before it becomes input to the
      forward model, you must install a spesific job for that.

   This is quite similar to the gen_data type (maybe they should be
   unified), but the latter is targeted at dynamic data. Observe that
   the gen_data_config object contains less information than most of
   the xxx_config objects.
*/
   


#define  DEBUG
#define  TARGET_TYPE GEN_PARAM
#include "enkf_debug.h"


struct gen_data_struct {
  DEBUG_DECLARE
  gen_data_config_type * config;      	      /* Thin config object - mainly contains filename for remote load */
  char                  * data;        	      /* Actual storage - will be casted to double or float on use. */
};




gen_data_config_type * gen_data_get_config(const gen_data_type * gen_data) { return gen_data->config; }


void gen_data_realloc_data(gen_data_type * gen_data) {
  int byte_size = gen_data_config_get_byte_size(gen_data->config);

  if (byte_size > 0)
    gen_data->data = util_realloc(gen_data->data , byte_size , __func__);
  else 
    gen_data->data = util_safe_free( gen_data->data );
    
}



gen_data_type * gen_data_alloc(const gen_data_config_type * config) {
  gen_data_type * gen_data = util_malloc(sizeof * gen_data, __func__);
  gen_data->config    = (gen_data_config_type *) config;
  gen_data->data      = NULL;
  gen_data_realloc_data(gen_data);
  DEBUG_ASSIGN(gen_data)
  return gen_data;
}


/**
   

*/

gen_data_type * gen_data_copyc(const gen_data_type * gen_data) {
  gen_data_type * copy = gen_data_alloc(gen_data->config);
  
  if (gen_data->data != NULL) {
    int byte_size = gen_data_config_get_byte_size( gen_data->config );
    copy->data = util_alloc_copy(gen_data->data , byte_size , __func__);
  }
  
  return copy;
}
  
  
void gen_data_free_data(gen_data_type * gen_data) {
  util_safe_free(gen_data->data);
  gen_data->data = NULL;
}



void gen_data_free(gen_data_type * gen_data) {
  gen_data_free_data(gen_data);
  free(gen_data);
}




/**
   Observe that this function writes parameter size to disk, that is
   special. The reason is that the config object does not know the
   size (on allocation).
*/

bool gen_data_fwrite(const gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  {
    int size      = gen_data_config_get_data_size(gen_data->config);
    int byte_size = gen_data_config_get_byte_size(gen_data->config);
    
    enkf_util_fwrite_target_type(stream , GEN_PARAM);
    util_fwrite_int(size , stream);
    util_fwrite_compressed(gen_data->data , byte_size , stream);
  }
  return true;
}


/* 
   Observe that this function manipulates memory directly. This should
   ideally be left to the enkf_node layer, but for this type the data
   size is determined at load time.
*/

void gen_data_fread(gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  {   
    int size;
    enkf_util_fread_assert_target_type(stream , GEN_PARAM);
    size = util_fread_int(stream);
    util_safe_free(gen_data->data);
    gen_data->data = util_fread_alloc_compressed(stream);
    gen_data_config_assert_size(gen_data->config , size , NULL);
  }
}




int gen_data_serialize(const gen_data_type *gen_data ,serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  ecl_type_enum ecl_type = gen_data_config_get_ecl_type(gen_data->config);
  const int data_size    = gen_data_config_get_data_size(gen_data->config);
  const int active_size  = gen_data_config_get_active_size(gen_data->config);
  const int *active_list = gen_data_config_get_active_list(gen_data->config);
  
  int elements_added = 0;
  if (data_size > 0) 
    elements_added = enkf_serialize(gen_data->data , data_size , ecl_type , active_size , active_list , serial_state ,serial_offset , serial_vector);
  return elements_added;
}


void gen_data_deserialize(gen_data_type * gen_data , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  ecl_type_enum ecl_type = gen_data_config_get_ecl_type(gen_data->config);
  const int data_size    = gen_data_config_get_data_size(gen_data->config);
  const int active_size  = gen_data_config_get_active_size(gen_data->config);
  const int *active_list = gen_data_config_get_active_list(gen_data->config);
  
  if (data_size > 0)
    enkf_deserialize(gen_data->data , data_size , ecl_type , active_size , active_list  , serial_state , serial_vector);
  
  /*
    gen_data_truncate(gen_data);
  */
}




/**
   This function initializes the parameter. This is based on loading a
   (ASCII) file. The name of the file is derived from a path_fmt
   instance owned by the config object. Observe that there is *NO*
   header information in this file. We just read floating point
   numbers from a formatted stream until we reach EOF.

   When the read is complete it is checked/verified with the config
   object that this file was as long as the others we have loaded for
   other members.
*/



void gen_data_initialize(gen_data_type * gen_data , int iens) {
  char * init_file 	 = gen_data_config_alloc_initfile(gen_data->config , iens);
  FILE * stream    	 = util_fopen(init_file , "r");
  ecl_type_enum ecl_type = gen_data_config_get_ecl_type(gen_data->config);
  int sizeof_ctype       = ecl_util_get_sizeof_ctype(ecl_type);
  int buffer_elements    = gen_data_config_get_data_size(gen_data->config);
  int current_size       = 0;
  int fscanf_return      = 1; /* To keep the compiler happy .*/
  char * buffer ;
  
  if (buffer_elements == 0)
    buffer_elements = 100;
  
  buffer = util_malloc( buffer_elements * sizeof_ctype , __func__);
  {
    do {
      if (ecl_type == ecl_float_type) {
	float  * float_buffer = (float *) buffer;
	fscanf_return = fscanf(stream , "%g" , &float_buffer[current_size]);
      } else if (ecl_type == ecl_double_type) {
	double  * double_buffer = (double *) buffer;
	fscanf_return = fscanf(stream , "%lg" , &double_buffer[current_size]);
      } else util_abort("%s: god dammit - internal error \n",__func__);
      
      if (fscanf_return == 1)
	current_size += 1;
      
      if (current_size == buffer_elements) {
	buffer_elements *= 2;
	buffer = util_realloc( buffer , buffer_elements * sizeof_ctype , __func__);
      }
    } while (fscanf_return == 1);
  }
  if (fscanf_return != EOF) 
    util_abort("%s: scanning of %s before EOF was reached -- fix your file.\n" , __func__ , init_file);
  
  gen_data_config_assert_size(gen_data->config , current_size , init_file);
  gen_data_realloc_data(gen_data);
  memcpy(gen_data->data , buffer , current_size * sizeof_ctype);
  
  free(buffer);
  fclose(stream);
  free(init_file);
}



void gen_data_ecl_write(const gen_data_type * gen_data , const char * eclfile , fortio_type * fortio) {
  DEBUG_ASSERT(gen_data)
  gen_data_config_ecl_write(gen_data->config , eclfile , gen_data->data);
}



/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/


VOID_ALLOC(gen_data)
VOID_FREE(gen_data)
VOID_FREE_DATA(gen_data)
VOID_REALLOC_DATA(gen_data)
VOID_FWRITE    (gen_data)
VOID_FREAD     (gen_data)
VOID_COPYC     (gen_data)
VOID_SERIALIZE(gen_data)
VOID_DESERIALIZE(gen_data)
VOID_INITIALIZE(gen_data)
VOID_ECL_WRITE(gen_data)

