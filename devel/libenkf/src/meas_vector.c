/**
   See the file README.obs for ducumentation of the varios datatypes
   involved with observations/measurement/+++.
*/
#include <stdlib.h>
#include <stdio.h>
#include <enkf_util.h>
#include <meas_vector.h>
#include <util.h>
#include <double_vector.h>

struct meas_vector_struct {
  int      target_size;        /* The size of the buffer will be reset to this size when use is finished. */
  double_vector_type * data;
}; 



void meas_vector_reset(meas_vector_type * meas_vector) {
  double_vector_reset( meas_vector->data );
  double_vector_resize( meas_vector->data , meas_vector->target_size );
}


meas_vector_type * meas_vector_alloc() {
  meas_vector_type * meas_vector = util_malloc(sizeof * meas_vector, __func__);
  meas_vector->target_size = 32;
  meas_vector->data = double_vector_alloc(0 ,0);
  meas_vector_reset(meas_vector);
  return meas_vector;
}



void meas_vector_add(meas_vector_type * meas_vector, double value) {
  double_vector_append( meas_vector->data , value );
}


int meas_vector_get_nrobs(const meas_vector_type * vector) {
  return double_vector_size( vector->data );
}




void meas_vector_free(meas_vector_type * meas_vector) {
  double_vector_free( meas_vector->data );
  free(meas_vector);
}


const double * meas_vector_get_data_ref(const meas_vector_type * vector) {
  return double_vector_get_const_ptr( vector->data );
}


