#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <enkf_state.h>
#include <util.h>
#include <multflt_config.h>
#include <multflt.h>
#include <enkf_util.h>
#include <math.h>
#include <scalar.h>


#define  DEBUG
#define  TARGET_TYPE MULTFLT
#include "enkf_debug.h"


GET_DATA_SIZE_HEADER(multflt);


struct multflt_struct {
  DEBUG_DECLARE
  const multflt_config_type *config;
  scalar_type               *scalar;
};

/*****************************************************************/

void multflt_free_data(multflt_type *multflt) {
  scalar_free_data(multflt->scalar);
}



void multflt_free(multflt_type *multflt) {
  scalar_free(multflt->scalar);
  free(multflt);
}



void multflt_realloc_data(multflt_type *multflt) {
  scalar_realloc_data(multflt->scalar);
}


void multflt_output_transform(const multflt_type * multflt) {
  scalar_transform(multflt->scalar);
}

void multflt_set_data(multflt_type * multflt , const double * data) {
  scalar_set_data(multflt->scalar , data);
}


void multflt_get_data(const multflt_type * multflt , double * data) {
  scalar_get_data(multflt->scalar , data);
}

void multflt_get_output_data(const multflt_type * multflt , double * output_data) {
  scalar_get_output_data(multflt->scalar , output_data);
}


const double * multflt_get_data_ref(const multflt_type * multflt) {
  return scalar_get_data_ref(multflt->scalar);
}


const double * multflt_get_output_ref(const multflt_type * multflt) {
  return scalar_get_output_ref(multflt->scalar);
}


multflt_type * multflt_alloc(const multflt_config_type * config) {
  multflt_type * multflt  = malloc(sizeof *multflt);
  multflt->config = config;
  multflt->scalar   = scalar_alloc(config->scalar_config); 
  DEBUG_ASSIGN(multflt)
  return multflt;
}


void multflt_clear(multflt_type * multflt) {
  scalar_clear(multflt->scalar);
}



multflt_type * multflt_copyc(const multflt_type *multflt) {
  multflt_type * new = multflt_alloc(multflt->config); 
  scalar_memcpy(new->scalar , multflt->scalar);
  return new; 
}



static void __multflt_ecl_write(const multflt_type * multflt, const char * eclfile , bool direct) {
  FILE * stream   = enkf_util_fopen_w(eclfile , __func__);
  {
    const multflt_config_type *config = multflt->config;
    const int data_size       = multflt_config_get_data_size(config);
    const char **fault_names  = (const char **) config->fault_names;
    const double *output_data = scalar_get_output_ref(multflt->scalar);
    int k;
    
    if (!direct) 
      multflt_output_transform(multflt);
    
    fprintf(stream , "MULTFLT\n");
    for (k=0; k < data_size; k++)
      fprintf(stream , " \'%s\'      %g  / \n",fault_names[k] , output_data[k]);
    fprintf(stream , "/");
  }
  
  fclose(stream);
}


void multflt_ecl_write(const multflt_type * multflt, const char * eclfile) {
  __multflt_ecl_write(multflt , eclfile , false);
}


void multflt_fwrite(const multflt_type *multflt , FILE * stream) {
  DEBUG_ASSERT(multflt);
  enkf_util_fwrite_target_type(stream , MULTFLT);
  scalar_stream_fwrite(multflt->scalar , stream);
}


void multflt_fread(multflt_type * multflt , FILE * stream) {
  DEBUG_ASSERT(multflt);
  enkf_util_fread_assert_target_type(stream , MULTFLT , __func__);
  scalar_stream_fread(multflt->scalar , stream);
}



void multflt_swapout(multflt_type * multflt , FILE * stream) {
  multflt_fwrite(multflt , stream);
  multflt_free_data(multflt);
}


void multflt_swapin(multflt_type * multflt , FILE * stream) {
  multflt_realloc_data(multflt);
  multflt_fread(multflt , stream);
}


void multflt_truncate(multflt_type * multflt) {
  DEBUG_ASSERT(multflt)
  scalar_truncate( multflt->scalar );  
}



void  multflt_sample(multflt_type *multflt) {
  DEBUG_ASSERT(multflt)
  scalar_sample(multflt->scalar);  
}



int multflt_serialize(const multflt_type *multflt , int internal_offset , size_t serial_data_size , double *serial_data , size_t ens_size , size_t offset, bool * complete) {
  DEBUG_ASSERT(multflt);
  return scalar_serialize(multflt->scalar , internal_offset , serial_data_size, serial_data , ens_size , offset , complete);
}


int multflt_deserialize(multflt_type *multflt , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  DEBUG_ASSERT(multflt);
  return scalar_deserialize(multflt->scalar , internal_offset , serial_size , serial_data , stride , offset);
}




multflt_type * multflt_alloc_mean(int ens_size , const multflt_type **multflt_ens) {
  int iens;
  multflt_type * avg_multflt = multflt_copyc(multflt_ens[0]);
  for (iens = 1; iens < ens_size; iens++) 
    multflt_iadd(avg_multflt , multflt_ens[iens]);
  multflt_iscale(avg_multflt , 1.0 / ens_size);
  return avg_multflt;
}



/*****************************************************************/


void multflt_TEST() {
  const char * config_file = "/tmp/multflt_config.txt";
  FILE * stream = util_fopen(config_file , "w");
  fprintf(stream , "North  0.00  1.00   0  TANH \n");
  fprintf(stream , "East   0.00  0.50   2  \n");
  fprintf(stream , "West   0.00  2.00   0  NULL \n");
  fclose(stream);
  
  {
    const int ens_size = 1000;
    char path[64];
    int iens;
    multflt_config_type  * config      = multflt_config_fscanf_alloc(config_file);
    multflt_type        ** multflt_ens = malloc(ens_size * sizeof * multflt_ens);
    
    for (iens = 0; iens < ens_size; iens++) {
      multflt_ens[iens] = multflt_alloc(config);
      multflt_sample(multflt_ens[iens]);
      sprintf(path , "/tmp/%04d/MULTZ.INC" , iens + 1);
      util_make_path(path);
      multflt_ecl_write(multflt_ens[iens] , path);
    }
  }
}							 


const char * multflt_get_name(const multflt_type * multflt, int fault_nr) {
  return  multflt_config_get_name(multflt->config , fault_nr);
}


/*
  Mathops not implemented ... 
*/

MATH_OPS_SCALAR(multflt);
VOID_ALLOC(multflt);
VOID_SWAPOUT(multflt);
VOID_SWAPIN(multflt);
VOID_SERIALIZE (multflt);
VOID_DESERIALIZE (multflt);
/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

VOID_ECL_WRITE (multflt)
VOID_FWRITE (multflt)
VOID_FREAD  (multflt)
VOID_COPYC  (multflt)
VOID_FUNC   (multflt_sample    , multflt_type)
VOID_FUNC   (multflt_free      , multflt_type)

