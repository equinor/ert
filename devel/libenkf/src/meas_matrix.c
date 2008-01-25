#include <stdlib.h>
#include <meas_matrix.h>
#include <string.h>
#include <meas_vector.h>
#include <util.h>
#include <analysis.h>

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



void meas_vector_allocS(const meas_matrix_type * matrix , double **_S) {
  double * S;
  int offset = 0;
  int iens, ens_stride , obs_stride;
  
  const int nrobs = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  analysis_set_stride(matrix->ens_size , nrobs , &ens_stride , &obs_stride);
  S  = util_malloc(nrobs * matrix->ens_size * sizeof * S , __func__);
  for (iens = 0; iens < matrix->ens_size; iens++) {
    const meas_vector_type * vector = matrix->meas_vectors[iens];
    if (nrobs != meas_vector_get_nrobs(vector)) {
      fprintf(stderr,"%s: fatal internal error - not all measurement vectors equally long - aborting \n",__func__);
      abort();
    }
    if (obs_stride == 1)
      memcpy(&S[offset] , meas_vector_get_data_ref(vector) , nrobs * sizeof * S);
    else {
      fprintf(stderr,"%s: code currently assumes obs_stride = 1 - aborting \n",__func__);
      abort();
    }
  }

  /* 
     Code written to facilitate the return of mean and standard
     deviation of S - currently not used.
  */
  {
    int   iobs;

    double * S1 = util_calloc(nrobs , sizeof * S1 , __func__);
    double * S2 = util_calloc(nrobs , sizeof * S2 , __func__);
    double * meanS;
    double * stdS;


    for (iens = 0; iens < matrix->ens_size; iens++) 
      for (iobs = 0; iobs < nrobs; iobs++) {
	int index = iens * ens_stride + iobs * obs_stride;
	S1[iobs] += S[index];
	S2[iobs] += S[index] * S[index];
      }

    for (iobs = 0; iobs < nrobs; iobs++) {
      S1[iobs] *= 1.0 / matrix->ens_size;
      S2[iobs] *= 1.0 / matrix->ens_size - S1[iobs] * S1[iobs];
    }
    meanS = S1;
    stdS  = S2;


    /*
      Subtracting the mean - the standard deviation is
      currently not used for anything.
    */

    for (iens = 0; iens < matrix->ens_size; iens++) 
      for (iobs = 0; iobs < nrobs; iobs++) {
	int index = iens * ens_stride + iobs * obs_stride;
	S[index] -= meanS[iobs];
      }
    
    free(S1);
    free(S2);
  }
  *_S = S;
}

