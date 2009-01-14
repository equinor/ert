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
#include <enkf_serialize.h>


GET_DATA_SIZE_HEADER(equil);

/*****************************************************************/


struct equil_struct {
  int                      __type_id;
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
  equil_type * equil    = util_malloc(sizeof *equil , __func__);
  equil->config         = config;
  equil->scalar         = scalar_alloc(equil_config_get_scalar_config(config));
  equil->__type_id      = EQUIL;
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



void equil_ecl_write(const equil_type * equil, const char * eclfile, fortio_type * fortio) {
  FILE * stream   = util_fopen(eclfile , "w");
  const double *woc , *goc;
  equil_output_transform(equil);
  equil_get_woc_goc_ref(equil , &woc , &goc);
  equil_config_ecl_write(equil->config , woc , goc , stream);
  fclose(stream);
}



bool equil_fwrite(const equil_type * equil, FILE * stream) {
  enkf_util_fwrite_target_type(stream , EQUIL);
  scalar_stream_fwrite(equil->scalar , stream);
  return true;
}


void equil_fread(equil_type * equil , FILE * stream) {
  enkf_util_fread_assert_target_type(stream , EQUIL);
  scalar_stream_fread(equil->scalar , stream);
}



bool equil_initialize(equil_type *equil, int iens) {
  scalar_sample(equil->scalar);
  return true;
}


void equil_free(equil_type *equil) {
  scalar_free(equil->scalar);  
  free(equil);
}


int equil_serialize(const equil_type *equil , serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  return scalar_serialize(equil->scalar , serial_state , serial_offset , serial_vector);
}


void equil_deserialize(equil_type *equil , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  scalar_deserialize(equil->scalar , serial_state , serial_vector);
}



/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

SAFE_CAST(equil , EQUIL);
MATH_OPS_SCALAR(equil);
VOID_ALLOC(equil);
VOID_SERIALIZE (equil);
VOID_DESERIALIZE (equil);
VOID_INITIALIZE(equil);
VOID_FREE_DATA(equil)
VOID_ECL_WRITE (equil)
VOID_FWRITE (equil)
VOID_FREAD  (equil)
VOID_COPYC     (equil)
VOID_FREE (equil)

     
