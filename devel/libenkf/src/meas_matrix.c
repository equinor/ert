#include <math.h>
#include <stdlib.h>
#include <meas_matrix.h>
#include <string.h>
#include <meas_vector.h>
#include <util.h>


struct meas_matrix_struct {
  int ens_size;
  meas_vector_type ** meas_vectors;
};



meas_matrix_type * meas_matrix_alloc(int ens_size) {
  meas_matrix_type * meas = malloc(sizeof * meas);
  if (ens_size <= 0) {
    fprintf(stderr,"%s: ens_size must be > 0 - aborting \n",__func__);
    abort();
  }
  
  meas->ens_size     = ens_size;
  meas->meas_vectors = malloc(ens_size * sizeof * meas->meas_vectors);
  {
    int i;
    for (i = 0; i < meas->ens_size; i++)
      meas->meas_vectors[i] = meas_vector_alloc();
  }
  return meas;
}


void meas_matrix_free(meas_matrix_type * matrix) {
  int i;
  for (i=0; i < matrix->ens_size; i++)
    meas_vector_free(matrix->meas_vectors[i]);
  free(matrix->meas_vectors);
  free(matrix);
}



void meas_matrix_add(meas_matrix_type * matrix , int iens , double value) {
  meas_vector_add(matrix->meas_vectors[iens] , value);
}


void meas_matrix_reset(meas_matrix_type * matrix) {
  int iens;
  for (iens = 0; iens < matrix->ens_size; iens++) 
    meas_vector_reset(matrix->meas_vectors[iens]);
}


meas_vector_type * meas_matrix_iget_vector(const meas_matrix_type * matrix , int iens) {
  if (iens >= 0 && iens < matrix->ens_size) 
    return matrix->meas_vectors[iens];
  else {
    fprintf(stderr,"%s - aborting \n",__func__);
    abort();
  }
}



void printf_matrix(const double *M , int ny , int nx , int stride_y , int stride_x, const char * name , const char * fmt) {

  int ix , iy;
  for (iy=0; iy < ny; iy++) {
    if (iy == ny / 2)
      printf("%s = ",name);
    else {
      int i;
      for (i=0; i < strlen(name) + 3; i++)
	printf(" ");
    }
    printf("|");
    for (ix = 0; ix < nx; ix++) {
      int index = iy * stride_y + ix * stride_x;
      printf(fmt , M[index]);
    }
    printf("|\n");
  }
}


/**
This function computes the ensemble mean and ensemble std of the
various observables. The data are returned (by reference) in two
vectors allocated in this function.
*/

void meas_matrix_allocS_stats(const meas_matrix_type * matrix, double **_meanS , double **_stdS) {
  const int nrobs = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  int   iobs , iens;
  double * S1 = util_malloc(nrobs * sizeof * S1 , __func__);
  double * S2 = util_malloc(nrobs * sizeof * S2 , __func__);
  
  
  for (iens = 0; iens < matrix->ens_size; iens++) {
    const meas_vector_type * vector = matrix->meas_vectors[iens];
    const double        * meas_data = meas_vector_get_data_ref(vector);

    for (iobs = 0; iobs < nrobs; iobs++) {
      S1[iobs] += meas_data[iobs];
      S2[iobs] += meas_data[iobs] * meas_data[iobs];
    }
  }

  for (iobs = 0; iobs < nrobs; iobs++) {
    S1[iobs] *= 1.0 / matrix->ens_size;
    S2[iobs]  = sqrt(S2[iobs] / matrix->ens_size - S1[iobs] * S1[iobs]);
  }
  
  *_meanS = S1;
  *_stdS  = S2;
}



/*
  Observe that this code does *NOT* subtract the ensemble
  mean from S. This is in contrast to the original Fortran
  code which did that.
*/


double * meas_matrix_allocS(const meas_matrix_type * matrix, int ens_stride , int obs_stride) {
  double * S;
  int iens ;
  const int nrobs = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  S  = util_malloc(nrobs * matrix->ens_size * sizeof * S , __func__);
  for (iens = 0; iens < matrix->ens_size; iens++) {
    const meas_vector_type * vector = matrix->meas_vectors[iens];
    if (nrobs != meas_vector_get_nrobs(vector)) {
      fprintf(stderr,"%s: fatal internal error - not all measurement vectors equally long - aborting \n",__func__);
      abort();
    }
    
    if (obs_stride == 1) {
      int offset = iens * nrobs;
      memcpy(&S[offset] , meas_vector_get_data_ref(vector) , nrobs * sizeof * S);
    } else {
      const double * meas_data = meas_vector_get_data_ref(vector);
      int iobs;
      for (iobs = 0; iobs < nrobs; iobs++) {
	int index = iobs * obs_stride + iens * ens_stride;
	S[index] = meas_data[iobs];
      }
    }
  }

  
  return S;
}

