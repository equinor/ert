#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <util.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <scalar_config.h>
#include <scalar.h>
#include <enkf_util.h>




/*****************************************************************/

GET_DATA_SIZE_HEADER(scalar);

struct scalar_struct {
  const scalar_config_type *config;
  double                  *data;
  double                  *output_data;
  bool                     output_valid;
};

/*****************************************************************/
void scalar_clear(scalar_type * scalar) {
  const int size = scalar_config_get_data_size(scalar->config);   
  int k;
  for (k = 0; k < size; k++)
    scalar->data[k] = 0.0;
}

void scalar_set_data(scalar_type * scalar , const double * data) {
  memcpy(scalar->data , data , scalar_config_get_data_size(scalar->config) * sizeof * data);
  scalar->output_valid = false;
}


void scalar_get_data(const scalar_type * scalar , double * data) {
  memcpy(data , scalar->data , scalar_config_get_data_size(scalar->config) * sizeof * data);
}

void scalar_get_output_data(const scalar_type * scalar , double * output_data) {
  memcpy(output_data , scalar->output_data , scalar_config_get_data_size(scalar->config) * sizeof * output_data);
}


void scalar_realloc_data(scalar_type *scalar) {
  scalar->data        = enkf_util_calloc(scalar_config_get_data_size(scalar->config) , sizeof *scalar->data        , __func__);
  scalar->output_data = enkf_util_calloc(scalar_config_get_data_size(scalar->config) , sizeof *scalar->output_data , __func__);
}


void scalar_free_data(scalar_type *scalar) {
  free(scalar->data);
  free(scalar->output_data);
  scalar->data        = NULL;
  scalar->output_data = NULL;
}


scalar_type * scalar_alloc(const scalar_config_type * scalar_config) {
  scalar_type * scalar  = malloc(sizeof *scalar);
  scalar->config = scalar_config;
  scalar->data        = NULL;
  scalar->output_data = NULL;
  scalar_realloc_data(scalar);
  return scalar;
}


void scalar_memcpy(scalar_type * new, const scalar_type * old) {
  int size = scalar_config_get_data_size(old->config);
  memcpy(new->data , old->data , size * sizeof *old->data);
  new->output_valid = false;
}



scalar_type * scalar_copyc(const scalar_type *scalar) {
  scalar_type * new = scalar_alloc(scalar->config);
  scalar_memcpy(new , scalar);
  return new;
}


void scalar_stream_fread(scalar_type * scalar , FILE * stream) {

  int  size;
  fread(&size , sizeof  size     , 1 , stream);
  enkf_util_fread(scalar->data , sizeof *scalar->data , size , stream , __func__);
  fclose(stream);
  scalar->output_valid = false;

}


void scalar_stream_fwrite(const scalar_type * scalar , FILE * stream) {

  const int data_size = scalar_config_get_data_size(scalar->config);
  fwrite(&data_size              ,   sizeof  data_size     , 1 , stream);
  enkf_util_fwrite(scalar->data    ,   sizeof *scalar->data    ,data_size , stream , __func__);
  
}


char * scalar_swapout(scalar_type * scalar , const char * path) {
  /*char * ensfile = util_alloc_full_path(path , scalar_config_get_ensfile_ref(scalar->config));
  scalar_fwrite(scalar , ensfile);
  scalar_free_data(scalar);
  return ensfile;
  */
  fprintf(stderr,"%s: not implemented - aborting \n",__func__);
  abort();
}



void scalar_swapin(scalar_type * scalar , const char *file) {
  /*
    scalar_realloc_data(scalar);
    scalar_fread(scalar  , file);
  */
  fprintf(stderr,"%s: not implemented - aborting \n",__func__);
  abort();
}




void scalar_sample(scalar_type *scalar) {
  const scalar_config_type *config    = scalar->config;
  const bool              *active    = config->active;
  const double            *std       = config->std;
  const double            *mean      = config->mean;
  const int                data_size = scalar_config_get_data_size(config);
  int i;
  
  for (i=0; i < data_size; i++) 
    if (active[i])
      scalar->data[i] = enkf_util_rand_normal(mean[i] , std[i]);
  
  scalar->output_valid = false;
}




void scalar_free(scalar_type *scalar) {
  scalar_free_data(scalar);
  free(scalar);
}



int scalar_deserialize(scalar_type * scalar , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  const scalar_config_type *config      = scalar->config;
  const bool              *active     = config->active;
  const int                data_size  = scalar_config_get_data_size(config);
  return enkf_util_deserialize(scalar->data , active , internal_offset , data_size , serial_size , serial_data , offset , stride);
}




int scalar_serialize(const scalar_type *scalar , int internal_offset , size_t serial_data_size ,  double *serial_data , size_t stride , size_t offset , bool *complete) {
  const scalar_config_type *config      = scalar->config;
  const bool              *active     = config->active;
  const int                data_size  = scalar_config_get_data_size(config);
  
  return enkf_util_serialize(scalar->data , active , internal_offset , data_size , serial_data , serial_data_size , offset , stride , complete);
}


void scalar_truncate(scalar_type * scalar) {
  scalar_config_truncate(scalar->config , scalar->data);
}


void scalar_transform(scalar_type * scalar) {
  scalar_config_transform(scalar->config , scalar->data , scalar->output_data);
  scalar->output_valid = true;
}

const double * scalar_get_output_ref(const scalar_type * scalar) { return scalar->output_data; }
const double * scalar_get_data_ref  (const scalar_type * scalar) { return scalar->data; }


/*
void scalar_iadd(void *void_arg , const void *void_delta) {                
  scalar_type *arg         = (scalar_type *)       void_arg;  	       
  const scalar_type *delta = (const scalar_type *) void_delta;	       
  const scalar_config_type *config = arg->config; 			       
  const int data_size = config->data_size;                      			       
  int i;                                              			       
  if (config != delta->config) {                                                 
    fprintf(stderr,"%s:two scalarz object have different config objects - aborting \n",__func__);
    abort();                                                                   
  }                                                                              
  for (i=0; i < data_size; i++) {                           			       
    arg->data[i]        += delta->data[i];
    arg->output_data[i] += delta->output_data[i];
  }
}


void scalar_iscale(void *void_arg , double scale_factor) {                
  scalar_type *arg                 = (scalar_type *)       void_arg;  	       
  const scalar_config_type *config = arg->config; 			       
  const int data_size            = config->data_size;                      			       
  int i;                                              			       

  for (i=0; i < data_size; i++) {                           			       
    arg->data[i]        *= scale_factor;
    arg->output_data[i] *= scale_factor;
  }
}
*/




/*****************************************************************/


void scalar_TEST() {
  return ;
}




MATH_OPS(scalar)

/*
VOID_ALLOC(scalar)
VOID_FREE(scalar)
VOID_FREE_DATA(scalar)
VOID_REALLOC_DATA(scalar)
VOID_ENS_WRITE (scalar)
VOID_ENS_READ  (scalar)
VOID_COPYC     (scalar)
VOID_SWAPIN(scalar)
VOID_SWAPOUT(scalar)
VOID_SERIALIZE(scalar)
VOID_TRUNCATE(scalar)



VOID_FUNC      (scalar_clear        , scalar_type)
VOID_FUNC      (scalar_sample       , scalar_type)
*/

