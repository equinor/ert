/**
   See the file README.obs for ducumentation of the varios datatypes
   involved with observations/measurement/+++.
*/

#include <math.h>
#include <stdlib.h>
#include <meas_matrix.h>
#include <string.h>
#include <meas_vector.h>
#include <util.h>
#include <matrix.h>

struct meas_matrix_struct {
  int ens_size;
  meas_vector_type ** meas_vectors;
  bool              * active;  
};



meas_matrix_type * meas_matrix_alloc(int ens_size) {
  meas_matrix_type * meas = malloc(sizeof * meas);
  if (ens_size <= 0) 
    util_abort("%s: ens_size must be > 0 - aborting \n",__func__);

  meas->ens_size     = ens_size;
  meas->meas_vectors = util_malloc(ens_size * sizeof * meas->meas_vectors , __func__);
  meas->active       = NULL;
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
  util_safe_free( matrix->active );
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
    util_abort("%s - aborting \n",__func__);
    return NULL; /* Compiler shut up */
  }
}



void fprintf_matrix(FILE * stream, const double *M, int ny, int nx, int stride_y, int stride_x, const char * name, const char * fmt) {

  int ix , iy;
  for (iy=0; iy < ny; iy++) {
    if (iy == ny / 2)
      fprintf(stream, "%s = ",name);
    else {
      int i;
      for (i=0; i < strlen(name) + 3; i++)
	fprintf(stream, " ");
    }
    fprintf(stream, "|");
    for (ix = 0; ix < nx; ix++) {
      int index = iy * stride_y + ix * stride_x;
      fprintf(stream, fmt , M[index]);
    }
    fprintf(stream, "|\n");
  }

}



void printf_matrix(const double *M , int ny , int nx , int stride_y , int stride_x, const char * name , const char * fmt) {
        fprintf_matrix(stdout, M, ny, nx, stride_y, stride_x, name, fmt);
}

void fwrite_matrix(const char * filename, const double *M , int ny , int nx , int stride_y , int stride_x, const char * name , const char * fmt) {
        FILE * stream = util_fopen(filename, "w");
        fprintf_matrix(stream, M, ny, nx, stride_y, stride_x, name, fmt);
        fclose(stream);
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
  
  for (iobs = 0; iobs < nrobs; iobs++) {
    S1[iobs] = 0;
    S2[iobs] = 0;
  }
  
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
    S2[iobs]  = sqrt( util_double_max(0 , S2[iobs] / matrix->ens_size - S1[iobs] * S1[iobs]));
  }
  
  *_meanS = S1;
  *_stdS  = S2;
}



 

matrix_type * meas_matrix_allocS__(const meas_matrix_type * matrix , int nrobs_active , double ** _meanS , const bool * active_obs) {
  matrix_type * S;
  double * meanS;
  int iens , active_iobs;
  const int nrobs_total = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  S     = matrix_alloc( nrobs_active , matrix->ens_size);
  meanS = util_malloc(nrobs_active * sizeof * meanS , __func__);
  
  for (active_iobs = 0; active_iobs < nrobs_active; active_iobs++)
    meanS[active_iobs] = 0;

  for (iens = 0; iens < matrix->ens_size; iens++) {
    const meas_vector_type * vector = matrix->meas_vectors[iens];
    if (nrobs_total != meas_vector_get_nrobs(vector)) 
      util_abort("%s: fatal internal error - not all measurement vectors equally long - aborting \n",__func__);
    
    {
      const double * meas_data = meas_vector_get_data_ref(vector);
      int total_iobs;
      active_iobs = 0;
      for (total_iobs = 0; total_iobs < nrobs_total; total_iobs++) {
	if (active_obs[total_iobs]) {
	  matrix_iset(S , active_iobs , iens , meas_data[total_iobs]);
	  meanS[active_iobs] += meas_data[total_iobs];
	  active_iobs++;
	}
      }
    }
  }

  /*
    Subtracting the (ensemble mean) of each measurement.
  */
  for (active_iobs = 0; active_iobs < nrobs_active; active_iobs++)
    meanS[active_iobs] /= matrix->ens_size;

  for (iens = 0; iens < matrix->ens_size; iens++) 
    for (active_iobs = 0; active_iobs < nrobs_active; active_iobs++) 
      matrix_iadd( S , active_iobs , iens , -meanS[active_iobs]);
  
  /** Let that leak for now. */
  // *_meanS = meanS;
      
  return S;
}


/**
   In the return value S - the mean value has been subtracted. _meanS
   is "returned by reference"
*/

double * meas_matrix_allocS(const meas_matrix_type * matrix, int nrobs_active , int ens_stride , int obs_stride, double ** _meanS , const bool * active_obs) {
  double * S;
  double * meanS;
  int iens , active_iobs;
  const int nrobs_total = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  S     = util_malloc(nrobs_active * matrix->ens_size * sizeof * S     , __func__);
  meanS = util_malloc(nrobs_active *                    sizeof * meanS , __func__);
  
  for (active_iobs = 0; active_iobs < nrobs_active; active_iobs++)
    meanS[active_iobs] = 0;

  for (iens = 0; iens < matrix->ens_size; iens++) {
    const meas_vector_type * vector = matrix->meas_vectors[iens];
    if (nrobs_total != meas_vector_get_nrobs(vector)) {
      fprintf(stderr,"%s: fatal internal error - not all measurement vectors equally long - aborting \n",__func__);
      abort();
    }
    
    {
      const double * meas_data = meas_vector_get_data_ref(vector);
      int total_iobs;
      active_iobs = 0;
      for (total_iobs = 0; total_iobs < nrobs_total; total_iobs++) {
	if (active_obs[total_iobs]) {
	  int index = active_iobs * obs_stride  +  iens * ens_stride;
	  S[index]            = meas_data[total_iobs];
	  meanS[active_iobs] += meas_data[total_iobs];
	  active_iobs++;
	}
      }
    }
  }

  /*
    Subtracting the (ensemble mean) of each measurement.
  */
  for (active_iobs = 0; active_iobs < nrobs_active; active_iobs++)
    meanS[active_iobs] /= matrix->ens_size;

  for (iens = 0; iens < matrix->ens_size; iens++) 
    for (active_iobs = 0; active_iobs < nrobs_active; active_iobs++) {
      int index = active_iobs * obs_stride  +  iens * ens_stride;
      S[index] -= meanS[active_iobs];
    }
  *_meanS = meanS;
      
  return S;
}

