#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <enkf_state.h>
#include <util.h>
#include <gen_kw_config.h>
#include <gen_kw.h>
#include <enkf_util.h>
#include <math.h>
#include <scalar.h>


#define  DEBUG
#define  TARGET_TYPE GEN_KW
#include "enkf_debug.h"


GET_DATA_SIZE_HEADER(gen_kw);


struct gen_kw_struct {
  DEBUG_DECLARE
  const gen_kw_config_type *config;
  scalar_type               *scalar;
};

/*****************************************************************/

void gen_kw_free_data(gen_kw_type *gen_kw) {
  scalar_free_data(gen_kw->scalar);
}



void gen_kw_free(gen_kw_type *gen_kw) {
  scalar_free(gen_kw->scalar);
  free(gen_kw);
}



void gen_kw_realloc_data(gen_kw_type *gen_kw) {
  scalar_realloc_data(gen_kw->scalar);
}


void gen_kw_output_transform(const gen_kw_type * gen_kw) {
  scalar_transform(gen_kw->scalar);
}

void gen_kw_set_data(gen_kw_type * gen_kw , const double * data) {
  scalar_set_data(gen_kw->scalar , data);
}


void gen_kw_get_data(const gen_kw_type * gen_kw , double * data) {
  scalar_get_data(gen_kw->scalar , data);
}

void gen_kw_get_output_data(const gen_kw_type * gen_kw , double * output_data) {
  scalar_get_output_data(gen_kw->scalar , output_data);
}


const double * gen_kw_get_data_ref(const gen_kw_type * gen_kw) {
  return scalar_get_data_ref(gen_kw->scalar);
}


const double * gen_kw_get_output_ref(const gen_kw_type * gen_kw) {
  return scalar_get_output_ref(gen_kw->scalar);
}


gen_kw_type * gen_kw_alloc(const gen_kw_config_type * config) {
  gen_kw_type * gen_kw  = malloc(sizeof *gen_kw);
  gen_kw->config = config;
  gen_kw->scalar   = scalar_alloc(config->scalar_config); 
  DEBUG_ASSIGN(gen_kw)
  return gen_kw;
}


void gen_kw_clear(gen_kw_type * gen_kw) {
  scalar_clear(gen_kw->scalar);
}



gen_kw_type * gen_kw_copyc(const gen_kw_type *gen_kw) {
  gen_kw_type * new = gen_kw_alloc(gen_kw->config); 
  scalar_memcpy(new->scalar , gen_kw->scalar);
  return new; 
}


void gen_kw_fwrite(const gen_kw_type *gen_kw , FILE * stream) {
  DEBUG_ASSERT(gen_kw);
  enkf_util_fwrite_target_type(stream , GEN_KW);
  scalar_stream_fwrite(gen_kw->scalar , stream);
}


void gen_kw_fread(gen_kw_type * gen_kw , FILE * stream) {
  DEBUG_ASSERT(gen_kw);
  enkf_util_fread_assert_target_type(stream , GEN_KW , __func__);
  scalar_stream_fread(gen_kw->scalar , stream);
}



void gen_kw_swapout(gen_kw_type * gen_kw , FILE * stream) {
  gen_kw_fwrite(gen_kw , stream);
  gen_kw_free_data(gen_kw);
}


void gen_kw_swapin(gen_kw_type * gen_kw , FILE * stream) {
  gen_kw_realloc_data(gen_kw);
  gen_kw_fread(gen_kw , stream);
}


void gen_kw_truncate(gen_kw_type * gen_kw) {
  DEBUG_ASSERT(gen_kw)
  scalar_truncate( gen_kw->scalar );  
}



void  gen_kw_sample(gen_kw_type *gen_kw) {
  DEBUG_ASSERT(gen_kw)
  scalar_sample(gen_kw->scalar);  
}



int gen_kw_serialize(const gen_kw_type *gen_kw , int internal_offset , size_t serial_data_size , double *serial_data , size_t ens_size , size_t offset, bool * complete) {
  DEBUG_ASSERT(gen_kw);
  return scalar_serialize(gen_kw->scalar , internal_offset , serial_data_size, serial_data , ens_size , offset , complete);
}


int gen_kw_deserialize(gen_kw_type *gen_kw , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  DEBUG_ASSERT(gen_kw);
  return scalar_deserialize(gen_kw->scalar , internal_offset , serial_size , serial_data , stride , offset);
}




gen_kw_type * gen_kw_alloc_mean(int ens_size , const gen_kw_type **gen_kw_ens) {
  int iens;
  gen_kw_type * avg_gen_kw = gen_kw_copyc(gen_kw_ens[0]);
  for (iens = 1; iens < ens_size; iens++) 
    gen_kw_iadd(avg_gen_kw , gen_kw_ens[iens]);
  gen_kw_iscale(avg_gen_kw , 1.0 / ens_size);
  return avg_gen_kw;
}


void gen_kw_filter_file(const gen_kw_type * gen_kw , const char * target_file) {
  const int size             = gen_kw_config_get_data_size(gen_kw->config);
  const double * output_data = scalar_get_output_ref(gen_kw->scalar);
  hash_type * kw_hash = hash_alloc(10);
  int ikw;

  gen_kw_output_transform(gen_kw);
  for (ikw = 0; ikw < size; ikw++)
    hash_insert_hash_owned_ref(kw_hash , gen_kw_config_get_name(gen_kw->config , ikw) , void_arg_alloc_double(output_data[ikw]) , void_arg_free__);
  util_filter_file(gen_kw_config_get_template_ref(gen_kw->config) , target_file , '<' , '>' , kw_hash);
  hash_free(kw_hash);

}

const char * gen_kw_get_name(const gen_kw_type * gen_kw, int kw_nr) {
  return  gen_kw_config_get_name(gen_kw->config , kw_nr);
}


/*
  Mathops not implemented ... 
*/

MATH_OPS_SCALAR(gen_kw);
VOID_ALLOC(gen_kw);
VOID_SWAPOUT(gen_kw);
VOID_SWAPIN(gen_kw);
VOID_SERIALIZE (gen_kw);
VOID_DESERIALIZE (gen_kw);
/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

VOID_FWRITE (gen_kw)
VOID_FREAD  (gen_kw)
VOID_COPYC  (gen_kw)
VOID_FUNC   (gen_kw_sample    , gen_kw_type)
VOID_FUNC   (gen_kw_free      , gen_kw_type)

