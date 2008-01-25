#include <stdlib.h>
#include <stdio.h>
#include <enkf_util.h>
#include <meas_vector.h>

struct meas_vector_struct {
  int     size;
  int     alloc_size;
  double *data;
}; 


static void meas_vector_realloc_data(meas_vector_type * meas_vector, int new_alloc_size) {
  meas_vector->alloc_size = new_alloc_size;
  meas_vector->data       = enkf_util_realloc(meas_vector->data , new_alloc_size * sizeof * meas_vector->data , __func__);
}


void meas_vector_reset(meas_vector_type * meas_vector) {
  meas_vector->size = 0;
}


meas_vector_type * meas_vector_alloc() {
  meas_vector_type * meas_vector = malloc(sizeof * meas_vector);
  meas_vector->data = NULL;
  meas_vector->alloc_size = 0;
  meas_vector_realloc_data(meas_vector , 10);
  meas_vector_reset(meas_vector);
  return meas_vector;
}



void meas_vector_add(meas_vector_type * meas_vector, double value) {
  if (meas_vector->size == meas_vector->alloc_size)
    meas_vector_realloc_data(meas_vector , 2*meas_vector->alloc_size + 2);
  meas_vector->data[meas_vector->size] = value;
  meas_vector->size++;
}


int meas_vector_get_nrobs(const meas_vector_type * vector) {
  return vector->size;
}


void meas_vector_fprintf(const meas_vector_type * meas_vector , FILE *stream) {
  int i;
  for (i = 0; i < meas_vector->size; i++)
    fprintf(stream , "%-3d : %12.3f\n", i+1 , meas_vector->data[i]);
}


void meas_vector_free(meas_vector_type * meas_vector) {
  free(meas_vector->data);
  free(meas_vector);
}


const double * meas_vector_get_data_ref(const meas_vector_type * vector) {
  return vector->data;
}

