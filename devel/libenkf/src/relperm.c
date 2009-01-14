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
#include <fortio.h>
#include <enkf_macros.h>


/*****************************************************************/
struct relperm_struct{
  int                          __type_id;
  const relperm_config_type  * config;
  scalar_type                * scalar;
};

/*****************************************************************/

void relperm_free(relperm_type * relperm){
  /* Need to check if correct type is sent from enkf_node.c */
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
  enkf_util_fwrite_target_type(stream , RELPERM);
  scalar_stream_fwrite(relperm->scalar , stream);
  return true;
}



void relperm_fread(relperm_type * relperm , FILE * stream) {
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
  relperm->__type_id = RELPERM;
  return relperm;
}

int relperm_serialize(const relperm_type *relperm , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  return scalar_serialize(relperm->scalar , serial_state , serial_offset , serial_vector);
}

void  relperm_deserialize(relperm_type *relperm , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  scalar_deserialize(relperm->scalar ,  serial_state , serial_vector);
}

void relperm_truncate(relperm_type * relperm) {
  scalar_truncate( relperm->scalar );  
}

void relperm_clear(relperm_type * relperm) {
  scalar_clear(relperm->scalar);
}

bool relperm_initialize(relperm_type * relperm, int iens) {
  scalar_sample(relperm->scalar);
  return true;
}


void relperm_get_data(const relperm_type * relperm, double * data) {
  scalar_get_data(relperm->scalar,data);
}

void relperm_set_data(const relperm_type * relperm, const double * data){
  scalar_set_data(relperm->scalar,data);
}

void relperm_ecl_write_f90test(const relperm_type * relperm, const double * data, const char * path){
  {
    relperm_set_data(relperm,data);
    relperm_output_transform(relperm);
    relperm_config_ecl_write_table(relperm->config, relperm_get_output_ref(relperm),path); 
  }
}

void relperm_ecl_write(const relperm_type * relperm , const char * __eclfile , fortio_type * fortio) {
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

SAFE_CAST(relperm , RELPERM)
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


