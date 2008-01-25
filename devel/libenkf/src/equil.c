#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <util.h>
#include <enkf_types.h>
#include <equil_config.h>
#include <equil.h>
#include <enkf_util.h>
#include <scalar.h>
#include <scalar_config.h>
#include <enkf_macros.h>


#define  DEBUG
#define  TARGET_TYPE EQUIL
#include "enkf_debug.h"

GET_DATA_SIZE_HEADER(equil);

/*****************************************************************/


struct equil_struct {
  DEBUG_DECLARE
  const equil_config_type *config;
  scalar_type             *scalar;
};

/*****************************************************************/

void equil_free_data(equil_type * equil) {
  scalar_free_data(equil->scalar);
}



void equil_realloc_data(equil_type * equil) {
  scalar_realloc_data(equil->scalar);
}


equil_type * equil_alloc(const equil_config_type * config) {
  equil_type * equil    = malloc(sizeof *equil);
  equil->config         = config;
  equil->scalar         = scalar_alloc(config->scalar_config);
  DEBUG_ASSIGN(equil)
  return equil;
}


equil_type * equil_copyc(const equil_type * src) {
  equil_type * new = equil_alloc(src->config);
  scalar_memcpy(new->scalar , src->scalar);
  return new;
}


void equil_output_transform(const equil_type * equil) {
  scalar_transform(equil->scalar);
}


static void equil_get_woc_goc_ref(const equil_type * equil, const double **woc , const double **goc) {
  const int data_size = equil_config_get_data_size(equil->config);
  const double * data = scalar_get_output_ref(equil->scalar);
  *woc = data;
  *goc = &data[data_size];
}



void equil_ecl_write(const equil_type * equil, const char * eclfile) {
  FILE * stream   = enkf_util_fopen_w(eclfile , __func__);
  const double *woc , *goc;
  equil_output_transform(equil);
  equil_get_woc_goc_ref(equil , &woc , &goc);
  equil_config_ecl_write(equil->config , woc , goc , stream);
  fclose(stream);
}



void equil_fwrite(const equil_type * equil, FILE * stream) {
  DEBUG_ASSERT(equil);
  enkf_util_fwrite_target_type(stream , EQUIL);
  scalar_stream_fwrite(equil->scalar , stream);
}

void equil_fread(equil_type * equil , FILE * stream) {
  DEBUG_ASSERT(equil);
  enkf_util_fread_assert_target_type(stream , EQUIL , __func__);
  scalar_stream_fread(equil->scalar , stream);
}


void equil_swapout(equil_type * equil , FILE * stream) {
  equil_fwrite(equil , stream);
  equil_free_data(equil);
}


void equil_swapin(equil_type * equil , FILE * stream) {
  equil_realloc_data(equil);
  equil_fread(equil , stream);
}




void equil_sample(equil_type *equil) {
  scalar_sample(equil->scalar);
}


void equil_free(equil_type *equil) {
  DEBUG_ASSERT(equil)
  scalar_free(equil->scalar);  
  free(equil);
}


int equil_serialize(const equil_type *equil , int internal_offset , size_t serial_data_size , double *serial_data , size_t stride , size_t offset, bool * complete) {
  DEBUG_ASSERT(equil);
  return scalar_serialize(equil->scalar , internal_offset , serial_data_size , serial_data , stride , offset , complete);
}

int equil_deserialize(equil_type *equil , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  DEBUG_ASSERT(equil);
  return scalar_deserialize(equil->scalar , internal_offset , serial_size , serial_data , stride , offset);
}




VOID_SWAPOUT(equil);
VOID_SWAPIN(equil);
MATH_OPS_SCALAR(equil);
VOID_ALLOC(equil);
VOID_SERIALIZE (equil);
VOID_DESERIALIZE (equil);
/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
VOID_ECL_WRITE (equil)
VOID_FWRITE (equil)
VOID_FREAD  (equil)
VOID_COPYC     (equil)
VOID_FUNC      (equil_sample    , equil_type)
VOID_FUNC      (equil_free      , equil_type)

