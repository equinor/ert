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
#include <gen_param_config.h>
#include <gen_param.h>




/**
   This file implements a "general parameter" type. The use of
   gen_param instances is roughly as follows:

   1. Some external program creates realizations of the parameter.

   2. The EnKF program loads the parameters in the initialization
      step; EnKF has by construction __NO_WAY__ to sample a gen_param
      instance.

   3. The parameter is updated by EnKF in the normal way, and then
      written out to the run_path directories. If you need fancy
      post-processing of the parameter before it becomes input to the
      forward model, you must install a spesific job for that.

   This is quite similar to the gen_data type (maybe they should be
   unified), but the latter is targeted at dynamic data. Observe that
   the gen_param_config object contains less information than most of
   the xxx_config objects.
*/
   


#define  DEBUG
#define  TARGET_TYPE GEN_PARAM
#include "enkf_debug.h"


struct gen_param_struct {
  DEBUG_DECLARE
  gen_param_config_type * config;      	      /* Thin config object - mainly contains filename for remote load */
  char                  * data;        	      /* Actual storage - will be casted to double or float on use. */
};




gen_param_config_type * gen_param_get_config(const gen_param_type * gen_param) { return gen_param->config; }


void gen_param_realloc_data(gen_param_type * gen_param) {
  int byte_size = gen_param_config_get_byte_size(gen_param->config);

  if (byte_size > 0)
    gen_param->data = util_realloc(gen_param->data , byte_size , __func__);
  else 
    gen_param->data = util_safe_free( gen_param->data );
    
}



gen_param_type * gen_param_alloc(const gen_param_config_type * config) {
  gen_param_type * gen_param = util_malloc(sizeof * gen_param, __func__);
  gen_param->config    = (gen_param_config_type *) config;
  gen_param->data      = NULL;
  gen_param_realloc_data(gen_param);
  DEBUG_ASSIGN(gen_param)
  return gen_param;
}


/**
   

*/

gen_param_type * gen_param_copyc(const gen_param_type * gen_param) {
  gen_param_type * copy = gen_param_alloc(gen_param->config);
  
  if (gen_param->data != NULL) {
    int byte_size = gen_param_config_get_byte_size( gen_param->config );
    copy->data = util_alloc_copy(gen_param->data , byte_size , __func__);
  }
  
  return copy;
}
  
  
void gen_param_free_data(gen_param_type * gen_param) {
  util_safe_free(gen_param->data);
  gen_param->data = NULL;
}



void gen_param_free(gen_param_type * gen_param) {
  gen_param_free_data(gen_param);
  free(gen_param);
}




/**
   Observe that this function writes parameter size to disk, that is
   special. The reason is that the config object does not know the
   size (on allocation).
*/

void gen_param_fwrite(const gen_param_type * gen_param , FILE * stream) {
  DEBUG_ASSERT(gen_param)
  {
    int size      = gen_param_config_get_data_size(gen_param->config);
    int byte_size = gen_param_config_get_byte_size(gen_param->config);
    
    enkf_util_fwrite_target_type(stream , GEN_PARAM);
    util_fwrite_int(size , stream);
    util_fwrite_compressed(gen_param->data , byte_size , stream);
  }
}


/* 
   Observe that this function manipulates memory directly. This should
   ideally be left to the enkf_node layer, but for this type the data
   size is determined at load time.
*/

void gen_param_fread(gen_param_type * gen_param , FILE * stream) {
  DEBUG_ASSERT(gen_param)
  {   
    int size;
    enkf_util_fread_assert_target_type(stream , GEN_PARAM);
    size = util_fread_int(stream);
    util_safe_free(gen_param->data);
    gen_param->data = util_fread_alloc_compressed(stream);
    gen_param_config_assert_size(gen_param->config , size , NULL);
  }
}




int gen_param_serialize(const gen_param_type *gen_param , int internal_offset , size_t serial_data_size ,  double *serial_data , size_t stride , size_t offset , bool *complete) {
  ecl_type_enum ecl_type = gen_param_config_get_ecl_type(gen_param->config);
  const int data_size    = gen_param_config_get_data_size(gen_param->config);
  const bool * iactive   = gen_param_config_get_iactive( gen_param->config );  
  int elements_added = 0;
  if (data_size > 0) 
    elements_added = enkf_util_serializeII(gen_param->data , ecl_type , iactive , internal_offset , data_size , serial_data , serial_data_size , offset , stride , complete);
  return elements_added;
}


int gen_param_deserialize(gen_param_type * gen_param , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  ecl_type_enum ecl_type = gen_param_config_get_ecl_type(gen_param->config);
  const int data_size    = gen_param_config_get_data_size(gen_param->config);
  const bool * iactive   = gen_param_config_get_iactive( gen_param->config );  
  int new_internal_offset = 0;
  
  if (data_size > 0)
    new_internal_offset = enkf_util_deserializeII(gen_param->data , ecl_type , iactive  , internal_offset , data_size , serial_size , serial_data , offset , stride);
  
  /*
    gen_param_truncate(gen_param);
  */
  return new_internal_offset;
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



void gen_param_initialize(gen_param_type * gen_param , int iens) {
  char * init_file 	 = gen_param_config_alloc_initfile(gen_param->config , iens);
  FILE * stream    	 = util_fopen(init_file , "r");
  ecl_type_enum ecl_type = gen_param_config_get_ecl_type(gen_param->config);
  int sizeof_ctype       = ecl_util_get_sizeof_ctype(ecl_type);
  int buffer_elements    = gen_param_config_get_data_size(gen_param->config);
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
  
  gen_param_config_assert_size(gen_param->config , current_size , init_file);
  gen_param_realloc_data(gen_param);
  memcpy(gen_param->data , buffer , current_size * sizeof_ctype);
  
  free(buffer);
  fclose(stream);
  free(init_file);
}



void gen_param_ecl_write(const gen_param_type * gen_param , const char * eclfile) {
  DEBUG_ASSERT(gen_param)
  gen_param_config_ecl_write(gen_param->config , eclfile , gen_param->data);
}



/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/


VOID_ALLOC(gen_param)
VOID_FREE(gen_param)
VOID_FREE_DATA(gen_param)
VOID_REALLOC_DATA(gen_param)
VOID_FWRITE    (gen_param)
VOID_FREAD     (gen_param)
VOID_COPYC     (gen_param)
VOID_SERIALIZE(gen_param)
VOID_DESERIALIZE(gen_param)
VOID_INITIALIZE(gen_param)
VOID_ECL_WRITE(gen_param)

     /*
       VOID_TRUNCATE(gen_param)
       VOID_SCALE(gen_param)
       ENSEMBLE_MULX_VECTOR(gen_param)
       ENSEMBLE_MULX_VECTOR_VOID(gen_param)
     */
