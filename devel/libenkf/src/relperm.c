#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <util.h>
#include <enkf_types.h>
#include <relperm_config.h>
#include <relperm.h>
#include <enkf_util.h>
#include <scalar.h>


#define  TARGET_TYPE RELPERM
#define  DEBUG
#include "enkf_debug.h"

/*****************************************************************/
struct relperm_struct{
  DEBUG_DECLARE
  const relperm_config_type  * config;
  scalar_type                * scalar;
};

/*****************************************************************/

void relperm_free(relperm_type * relperm){
  /* Need to check if correct type is sent from enkf_node.c */
  DEBUG_ASSERT(relperm)
    {
      scalar_free(relperm->scalar);
      free(relperm);
    }
}

void relperm_free_data(relperm_type * relperm){
  scalar_free_data(relperm->scalar);
}

void relperm_realloc_data(relperm_type * relperm){
  scalar_realloc_data(relperm->scalar);
}

bool relperm_fwrite(const relperm_type * relperm, FILE * stream){
  DEBUG_ASSERT(relperm);
  enkf_util_fwrite_target_type(stream , RELPERM);
  scalar_stream_fwrite(relperm->scalar , stream);
  return true;
}



void relperm_fread(relperm_type * relperm , FILE * stream) {
  DEBUG_ASSERT(relperm); 
  enkf_util_fread_assert_target_type(stream , RELPERM);
  scalar_stream_fread(relperm->scalar , stream);
}

relperm_type * relperm_copyc(const relperm_type *relperm) {
  relperm_type * new = relperm_alloc(relperm->config); 
  scalar_memcpy(new->scalar , relperm->scalar);
  return new; 
}

relperm_type * relperm_alloc(const relperm_config_type * relperm_config){
  relperm_type * relperm = malloc(sizeof *relperm);
  relperm->config = relperm_config;
  relperm->scalar = scalar_alloc(relperm_config->scalar_config);
  
  DEBUG_ASSIGN(relperm);
  return relperm;
}

int relperm_serialize(const relperm_type *relperm , int internal_offset , size_t serial_data_size , double *serial_data , size_t stride , size_t offset, bool * complete, serial_state_type * serial_state) {
  DEBUG_ASSERT(relperm);
  return scalar_serialize(relperm->scalar , internal_offset , serial_data_size , serial_data , stride , offset , complete , serial_state);
}

int relperm_deserialize(relperm_type *relperm , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset, serial_state_type * serial_state) {
  DEBUG_ASSERT(relperm);
  return scalar_deserialize(relperm->scalar , internal_offset , serial_size , serial_data , stride , offset , serial_state);
}

void relperm_truncate(relperm_type * relperm) {
  DEBUG_ASSERT(relperm)
  scalar_truncate( relperm->scalar );  
}

void relperm_clear(relperm_type * relperm) {
  scalar_clear(relperm->scalar);
}

void relperm_initialize(relperm_type * relperm, int iens) {
  DEBUG_ASSERT(relperm)
  scalar_sample(relperm->scalar);

}

void relperm_get_data(const relperm_type * relperm, double * data) {
  scalar_get_data(relperm->scalar,data);
}

void relperm_set_data(const relperm_type * relperm, const double * data){
  scalar_set_data(relperm->scalar,data);
}

void relperm_ecl_write_f90test(const relperm_type * relperm, const double * data, const char * path){
  DEBUG_ASSERT(relperm)
    {
      relperm_set_data(relperm,data);
      relperm_output_transform(relperm);
      relperm_config_ecl_write_table(relperm->config, relperm_get_output_ref(relperm),path); 
    }
}

void relperm_ecl_write(const relperm_type * relperm , const char * __eclfile) {
  DEBUG_ASSERT(relperm)
  {
    printf("Er i relperm_ecl_write_1 \n");
    char * eclfile;
    char * eclpath; 
    FILE * stream  = util_fopen(__eclfile , "w");
    relperm_output_transform(relperm);
    util_alloc_file_components(__eclfile, &eclpath,&eclfile, NULL);
    printf("Er i relperm_ecl_write_2 \n");
    relperm_config_ecl_write(relperm->config , relperm_get_output_ref(relperm) , stream,eclpath);
    fclose(stream);
  }
}

void relperm_output_transform(const relperm_type * relperm){
  scalar_transform(relperm->scalar);
}

const double * relperm_get_output_ref(const relperm_type * relperm){
  return scalar_get_output_ref(relperm->scalar) ;
}

void relperm_get_output_data(const relperm_type * relperm, double * output_data){
  scalar_get_output_data(relperm->scalar,output_data);
}

/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

MATH_OPS_SCALAR(relperm)
VOID_ALLOC(relperm)
VOID_FREE(relperm)
VOID_FREE_DATA(relperm)
VOID_REALLOC_DATA(relperm)
VOID_ECL_WRITE (relperm)
VOID_FWRITE    (relperm)
VOID_FREAD     (relperm)
VOID_COPYC     (relperm)
VOID_SERIALIZE(relperm)
VOID_DESERIALIZE(relperm)
VOID_TRUNCATE(relperm)
VOID_SCALE(relperm)
ENSEMBLE_MULX_VECTOR(relperm)
ENSEMBLE_MULX_VECTOR_VOID(relperm)
VOID_INITIALIZE(relperm)


