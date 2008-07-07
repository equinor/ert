#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <ecl_sum.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <fortio.h>
#include <ecl_util.h>
#include <gen_data_config.h>
#include <gen_data.h>
#include <gen_common.h>
#include <enkf_macros.h>



/**
   This file implements a "general data type"; the data type is
   originally devised to load and organise seismic data from a forward
   simulation.

   In contrast to the other enkf data types this type stores quite a
   lot of configuration information in the actual data object, and not
   in a config object. That is because much config information, like
   e.g. the length+++ is determined at run time when we load data
   produced by an external application.

   The same gen_data object can (in principle) be used in different
   ways through one simulation; however only one way at one time step.
*/
   


#define  DEBUG
#define  TARGET_TYPE GEN_DATA
#include "enkf_debug.h"


struct gen_data_struct {
  DEBUG_DECLARE
  gen_data_config_type         * config;      	      /* Thin config object - mainly contains filename for remote load */
  ecl_type_enum                  ecl_type;    	      /* Type of data can be ecl_float_type || ecl_double_type - read at load time, 
					      	         can change from time-step to time-step.*/           
  int   			 size;        	      /* The number of elements. */
  bool  			 active;      	      /* Is the keyword currently active ?*/
  char                         * data;        	      /* Actual storage - will be casted to double or float on use. */

  /*-----------------------------------------------------------------*/
  /* The variables below this line are only relevant when we are
     loading results from a forward simulation, and not actually part
     of the gen_data structure as such.
  */

  char                         * file_tag;     	      /* A tag written by the external software which identifies the data - can be NULL */ 
  gen_data_file_type             file_type;   	      /* The type of file this is. */ 
  bool                           fortran_endian_flip; /* IFF file_type == binary_fortran_file this variable is correct - otherwise rubbish. */
  char                         * src_file;            /* Name of the src_file we are currently trying to load from. */
};




gen_data_config_type * gen_data_get_config(const gen_data_type * gen_data) { return gen_data->config; }
int                     gen_data_get_size(const gen_data_type * gen_data)  { return gen_data->size; }


static void gen_data_fread_header(gen_data_type * gen_data , const char * config_tag , FILE * stream) {
  gen_common_fload_header(gen_data->file_type , stream , config_tag , &gen_data->file_tag , &gen_data->size , &gen_data->ecl_type);
}


static void gen_data_fread_data(gen_data_type * gen_data , FILE * stream) {
  gen_common_fload_data(stream , gen_data->src_file , gen_data->file_type , gen_data->ecl_type , gen_data->size , gen_data->data);
}


static void gen_data_set_file_data(gen_data_type * gen_data , const char * filename ) {
  gen_data->src_file = util_safe_free( gen_data->src_file );
  gen_data->file_tag = util_safe_free( gen_data->src_file );  
  
  gen_data->src_file = util_alloc_string_copy( filename );
  gen_common_get_file_type(filename , &gen_data->file_type , &gen_data->fortran_endian_flip);
}



gen_data_type * gen_data_alloc(const gen_data_config_type * config) {
  gen_data_type * gen_data = util_malloc(sizeof * gen_data, __func__);
  gen_data->config    = (gen_data_config_type *) config;
  gen_data->data      = NULL;
  gen_data->file_tag  = NULL;
  gen_data->src_file  = NULL;
  gen_data->size      = 0;
  gen_data->active    = false;
  DEBUG_ASSIGN(gen_data)
  return gen_data;
}


/**
 */

gen_data_type * gen_data_copyc(const gen_data_type * gen_data) {
  gen_data_type * copy = gen_data_alloc(gen_data->config);
  if (gen_data->active) {
    copy->active = true;
    copy->size     = gen_data->size;
    copy->ecl_type = gen_data->ecl_type;
    copy->data     = util_alloc_copy(gen_data->data , copy->size * ecl_util_get_sizeof_ctype(copy->ecl_type) , __func__);
  }
  return copy;
}
  
  
void gen_data_free_data(gen_data_type * gen_data) {
  util_safe_free(gen_data->data);
  gen_data->data = NULL;
}



void gen_data_free(gen_data_type * gen_data) {
  gen_data_free_data(gen_data);
  util_safe_free(gen_data->src_file);
  util_safe_free(gen_data->file_tag);
  free(gen_data);
}


void gen_data_realloc_data(gen_data_type * gen_data) {
  if (gen_data->size > 0)
    gen_data->data = util_realloc(gen_data->data , gen_data->size * ecl_util_get_sizeof_ctype(gen_data->ecl_type) , __func__);
}



/**
   Would prefer that this function was not called at all if the
   gen_data keyword does not (currently) hold data. The current
   implementation will leave a very small file without data.
*/

void gen_data_fwrite(const gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  enkf_util_fwrite_target_type(stream , GEN_DATA);
  util_fwrite_bool(gen_data->active , stream);
  if (gen_data->active) {
    util_fwrite_int(gen_data->size , stream);
    util_fwrite_int(gen_data->ecl_type , stream);
    util_fwrite_compressed(gen_data->data , gen_data->size * ecl_util_get_sizeof_ctype(gen_data->ecl_type) , stream);
  }
}


void gen_data_fread(gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  enkf_util_fread_assert_target_type(stream , GEN_DATA , __func__);
  gen_data->active = util_fread_bool(stream);
  if (gen_data->active) {
    gen_data->size = util_fread_int(stream);
    gen_data->size = util_fread_int(stream);
    gen_data_realloc_data(gen_data);
    util_fread_compressed(gen_data->data , stream);
  }
}


static void gen_data_deactivate(gen_data_type * gen_data) {
  if (gen_data->active) {
    gen_data->active = false;
    gen_data->size   = 0;
    gen_data_free_data( gen_data );
    gen_data_config_deactivate_metadata(gen_data->config);
  }
}


void gen_data_fload(gen_data_type * gen_data , const char * config_tag , const char * filename) {
  FILE * stream;
  gen_data_set_file_data(gen_data , filename);
  stream = util_fopen(filename , "r");
  gen_data_fread_header(gen_data , config_tag , stream);
  gen_data_realloc_data(gen_data);
  gen_data_fread_data(gen_data , stream);
  fclose(stream);
}


void gen_data_ecl_load(gen_data_type * gen_data , const char * run_path , const char * ecl_base , const ecl_sum_type * ecl_sum , int report_step) {
  DEBUG_ASSERT(gen_data)
  {
    gen_data_config_type * config = gen_data->config;
    
    if (gen_data_config_is_active(config , report_step)) {
      char *ecl_file;
      char *config_tag;
      char *full_path;
      gen_data_config_get_ecl_file(config , report_step , &ecl_file , &config_tag);
      full_path = util_alloc_full_path(run_path , ecl_file);
      
      if (util_file_exists(full_path)) {
	gen_data_fload(gen_data , config_tag , full_path);
	gen_data_config_assert_metadata(gen_data->config , report_step , gen_data->size , gen_data->ecl_type , gen_data->file_tag);
      } else 
	util_abort("%s: At report_step:%d could not find file:%s.\n",__func__ , report_step , full_path);

      if (gen_data->size == 0)
	gen_data_deactivate(gen_data);
  
      free(full_path);
    } else
      gen_data_deactivate(gen_data);
  }
}


int gen_data_serialize(const gen_data_type *gen_data , int internal_offset , size_t serial_data_size ,  double *serial_data , size_t stride , size_t offset , bool *complete) {
  ecl_type_enum ecl_type = gen_data->ecl_type;
  const int data_size    = gen_data->size;
  int elements_added = 0;
  if (data_size > 0) 
    elements_added = enkf_util_serializeII(gen_data->data , ecl_type , NULL , internal_offset , data_size , serial_data , serial_data_size , offset , stride , complete);
  return elements_added;
}


int gen_data_deserialize(gen_data_type * gen_data , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  ecl_type_enum ecl_type  = gen_data->ecl_type;
  const int data_size     = gen_data->size;
  int new_internal_offset = 0;
  
  if (data_size > 0)
    new_internal_offset = enkf_util_deserializeII(gen_data->data , ecl_type , NULL , internal_offset , data_size , serial_size , serial_data , offset , stride);
  
  /*
    gen_data_truncate(gen_data);
  */
  return new_internal_offset;
}



/**
   Returns double value of element nr i in the gen_data instance. If
   underlying data_type is float, it is converted.
*/


double gen_data_iget_double(const gen_data_type * gen_data, int index) {
  return gen_common_iget_double(index, gen_data->size , gen_data->ecl_type , gen_data->data);
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
VOID_ECL_LOAD(gen_data)
VOID_SERIALIZE(gen_data)
VOID_DESERIALIZE(gen_data)
     
     /*
       VOID_TRUNCATE(gen_data)
       VOID_SCALE(gen_data)
       ENSEMBLE_MULX_VECTOR(gen_data)
       ENSEMBLE_MULX_VECTOR_VOID(gen_data)
       VOID_INITIALIZE(gen_data)
     */
