#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <util.h>
#include <enkf_types.h>
#include <scalar.h>
#include <enkf_util.h>
#include <enkf_serialize.h>
#include <buffer.h>


/*****************************************************************/

//GET_DATA_SIZE_HEADER(scalar);

struct scalar_struct {
  int                        size;
  double                   * data;
  double                   * output_data;
  bool                       output_valid;
  bool                       __output_locked;
};

/*****************************************************************/

void scalar_clear(scalar_type * scalar) {
  const int data_size = scalar->size;
  int k;
  for (k = 0; k < data_size; k++) {
    scalar->output_data[k] = 0.0;
    scalar->data[k]        = 0.0;
  }
}

void scalar_set_data(scalar_type * scalar , const double * data) {
  const int data_size = scalar->size;
  memcpy(scalar->data , data , data_size * sizeof * data);
  scalar->output_valid = false;
}


void scalar_get_data(const scalar_type * scalar , double * data) {
  const int data_size = scalar->size;
  memcpy(data , scalar->data , data_size * sizeof * data);
}



double scalar_iget_double(scalar_type * scalar , bool internal_value , int index) {
  util_abort("%s: Broken \n",__func__);
  //if (internal_value)
  //  return scalar->data[index];
  //else {
  //  if (!scalar->output_valid)
  //    scalar_transform( scalar );
  //  return scalar->output_data[index];
  //}
  return -1;
}


void scalar_iset(scalar_type * scalar , int index , double value) {
  scalar->data[index]  = value;
  scalar->output_valid = false;
}



void scalar_get_output_data(const scalar_type * scalar , double * output_data) {
  const int data_size = scalar->size;
  memcpy(output_data , scalar->output_data , data_size * sizeof * output_data);
}


void scalar_resize(scalar_type * scalar , int size) {
  scalar->size        = size;
  scalar->data        = util_malloc(size * sizeof *scalar->data        , __func__);
  scalar->output_data = util_malloc(size * sizeof *scalar->output_data , __func__);
}


void scalar_free_data(scalar_type *scalar) {
  free(scalar->data);
  free(scalar->output_data);
  scalar->data        = NULL;
  scalar->output_data = NULL;
}


scalar_type * scalar_alloc( int size ) {
  scalar_type * scalar  = malloc(sizeof *scalar);
  scalar->data        	  = NULL;
  scalar->output_data 	  = NULL;
  scalar->__output_locked = false;
  scalar_resize(scalar , size);
  return scalar;
}


void scalar_memcpy(scalar_type * new, const scalar_type * old) {
  int size = old->size;

  memcpy(new->data        , old->data        , size * sizeof *old->data);
  memcpy(new->output_data , old->output_data , size * sizeof *old->output_data);

  new->output_valid = old->output_valid;
}



scalar_type * scalar_copyc(const scalar_type *scalar) {
  scalar_type * new = scalar_alloc( scalar->size );
  scalar_memcpy(new , scalar);
  return new;
}



void scalar_stream_fread(scalar_type * scalar , FILE * stream) {

  int  size;
  fread(&size , sizeof  size     , 1 , stream);
  util_fread(scalar->data , sizeof *scalar->data , size , stream , __func__);
  scalar->output_valid = false;
  
}


void scalar_buffer_fload(scalar_type * scalar , buffer_type * buffer) {
  int size = scalar->size;
  buffer_fread(buffer , scalar->data , sizeof *scalar->data , size);
  scalar->output_valid = false;
}


void scalar_stream_fwrite(const scalar_type * scalar , FILE * stream , bool internal_state) {
  
  const int data_size = scalar->size;
  fwrite(&data_size     ,   sizeof  data_size     , 1 , stream);
  util_fwrite(scalar->data , sizeof *scalar->data    ,data_size , stream , __func__);

}


void scalar_buffer_fsave(const scalar_type * scalar , buffer_type * buffer , bool internal_state) {
  
  const int data_size = scalar->size;
  buffer_fwrite(buffer , scalar->data , sizeof *scalar->data    ,data_size);

}




void scalar_sample(scalar_type *scalar) {
  const double mean = 0.0; /* Mean and std are hardcoded - the variability should be in the transformation. */
  const double std  = 1.0; 
  const int                data_size = scalar->size;
  int i;
  
  for (i=0; i < data_size; i++) 
    scalar->data[i] = enkf_util_rand_normal(mean , std);
  
  scalar->output_valid = false;
}




void scalar_free(scalar_type *scalar) {
  scalar_free_data(scalar);
  free(scalar);
}






void scalar_matrix_deserialize(scalar_type * scalar , const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {
  const int                data_size    = scalar->size;
  enkf_matrix_deserialize( scalar->data , data_size , ECL_DOUBLE_TYPE , active_list , A , row_offset , column);
}


void scalar_matrix_serialize(const scalar_type *scalar ,  const active_list_type * active_list , matrix_type * A , int row_offset , int column) {
  const int                data_size    = scalar->size;
  enkf_matrix_serialize( scalar->data , data_size , ECL_DOUBLE_TYPE , active_list , A , row_offset , column);
}




void scalar_truncate(scalar_type * scalar) {
  return ; 
}



const double * scalar_get_output_ref(const scalar_type * scalar) { return scalar->output_data; }
      double * scalar_get_data_ref  (const scalar_type * scalar) { return scalar->data; }



static void scalar_lock_output(scalar_type * scalar) { scalar->__output_locked   = true; }
//static void scalar_unlock_output(scalar_type * scalar) { scalar->__output_locked = false; }

/**
   The scalar object must implement it's own mathematical functions,
   because what we are interested in is the average of e.g. a
   multipier, and *NOT* of the underlying GAUSSIAN variable which is
   in scalar->data; and since transformations are generally
   non-linear, the mathematical operations, and the output transform
   do not commute.

   Trying to call output_transform on a scalar which has been manipulated
   with these functions will result in a fatal error.
*/

// Changed all thes functions to work on the scalar->data and NOT the output data - as of 02.08.2009.



void scalar_isqrt(scalar_type * scalar) {
  const int data_size = scalar->size;
  int i;
  
  for (i=0; i < data_size; i++) 
    scalar->data[i] = sqrt( scalar->data[i] );

}


void scalar_scale(scalar_type * scalar, double factor) {
  const int data_size = scalar->size;
  int i;
  for (i=0; i < data_size; i++) 
    scalar->data[i] *= factor;
}


void scalar_iadd(scalar_type * scalar , const scalar_type * delta) {
  const int data_size = scalar->size;
  int i;                                              			       
  for (i=0; i < data_size; i++) 
    scalar->data[i] += delta->data[i];
}


//void scalar_isub(scalar_type * scalar , const scalar_type * delta) {
//  const scalar_config_type *config = scalar->config; 			       
//  const int data_size = scalar_config_get_data_size(config);
//  int i;                                              			       
//  if (config != delta->config) util_abort("%s:two scalar object have different config objects - aborting \n",__func__);
//  scalar_lock_output(scalar);
//  for (i=0; i < data_size; i++) 
//    scalar->output_data[i] -= delta->output_data[i];
//}


void scalar_imul(scalar_type * scalar , const scalar_type * delta) {
  const int data_size = scalar->size;
  int i;                                              			       
  for (i=0; i < data_size; i++) 
    scalar->data[i] *= delta->data[i];
}


//void scalar_imul_add(scalar_type * scalar , double scale_factor , const scalar_type * delta) {
//  const scalar_config_type *config = scalar->config; 			       
//  const int data_size = scalar_config_get_data_size(config);
//  int i;                                              			       
//  if (config != delta->config) util_abort("%s:two scalar object have different config objects - aborting \n",__func__);
//  scalar_lock_output(scalar);
//  for (i=0; i < data_size; i++) 
//    scalar->output_data[i] += scale_factor * delta->output_data[i];
//}


void scalar_iaddsqr(scalar_type * scalar , const scalar_type * delta) {
  const int data_size = scalar->size;
  int i;                                              			       
  for (i=0; i < data_size; i++) 
    scalar->data[i] += delta->data[i] * delta->data[i];
}



/**
   Operates on the underlying normally distributed variable.
*/

void scalar_set_inflation(scalar_type * inflation , const scalar_type * std , const scalar_type * min_std) {
  const int data_size = std->size;
  for (int i=0; i < data_size; i++) {
    if (std->data[i] > 0)
      inflation->data[i] = util_double_max( 1.0 , min_std->data[i] / std->data[i]);   
    else
      inflation->data[i] = 1;
  }
}

