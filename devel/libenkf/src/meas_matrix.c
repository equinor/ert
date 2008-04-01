#include <stdlib.h>
#include <meas_matrix.h>
#include <string.h>
#include <meas_vector.h>
#include <util.h>


struct meas_matrix_struct {
  int ens_size;
  meas_vector_type ** meas_vectors;
  bool              * obs_active;
  bool                locked;
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
  meas->obs_active = NULL;
  return meas;
}


void meas_matrix_free(meas_matrix_type * matrix) {
  int i;
  for (i=0; i < matrix->ens_size; i++)
    meas_vector_free(matrix->meas_vectors[i]);
  if (matrix->obs_active != NULL)
    free(matrix->obs_active);
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
  free(matrix->obs_active);
  matrix->locked = false;
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


void meas_matrix_set_active(meas_matrix_type * matrix) {
  /*
    If all the ensemble members have the same value for an observation,
    it must be deactivated.
  */
  const int nrobs = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  int iobs,iens;
  matrix->obs_active = util_realloc(matrix->obs_active , nrobs * sizeof * matrix->obs_active , __func__);
  {
    for (iobs = 0; iobs < nrobs; iobs++) 
      matrix->obs_active[iobs] = false;
    
    meas_vector_type * vector2 = matrix->meas_vectors[0];
    meas_vector_type * vector1;
    for (iens = 1; iens < matrix->ens_size; iens++) {
      vector1 = vector2;
      vector2 = matrix->meas_vectors[iens];
      {
	const double * data1 = meas_vector_get_data_ref(vector1);
	const double * data2 = meas_vector_get_data_ref(vector2);
	for (iobs = 0; iobs < nrobs; iobs++) {
	  printf("Comparing: iens:%d-%d   iobs:%d %g %g \n",iens-1,iens,iobs,data1[iobs] , data2[iobs]);
	  if (data1[iobs] != data2[iobs]) matrix->obs_active[iobs] = true;
	}
      }
    }
    for (iobs = 0; iobs < nrobs; iobs++) 
      printf("Active:%d \n",matrix->obs_active[iobs]);
  }
  matrix->locked= true;
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
/*   /\*  */
/*      Code written to facilitate the return of mean and standard */
/*      deviation of S - currently not used. */
/*   *\/ */
/*   { */
/*     int   iobs; */

/*     double * S1 = util_calloc(nrobs , sizeof * S1 , __func__); */
/*     double * S2 = util_calloc(nrobs , sizeof * S2 , __func__); */
/*     double * meanS; */
/*     double * stdS; */


/*     for (iens = 0; iens < matrix->ens_size; iens++)  */
/*       for (iobs = 0; iobs < nrobs; iobs++) { */
/* 	int index = iens * ens_stride + iobs * obs_stride; */
/* 	S1[iobs] += S[index]; */
/* 	S2[iobs] += S[index] * S[index]; */
/*       } */

/*     for (iobs = 0; iobs < nrobs; iobs++) { */
/*       S1[iobs] *= 1.0 / matrix->ens_size; */
/*       S2[iobs] *= 1.0 / matrix->ens_size - S1[iobs] * S1[iobs]; */
/*     } */
/*     meanS = S1; */
/*     stdS  = S2; */


/*     /\* */
/*       Subtracting the mean - the standard deviation is */
/*       currently not used for anything. */
/*     *\/ */

/*     for (iens = 0; iens < matrix->ens_size; iens++)  */
/*       for (iobs = 0; iobs < nrobs; iobs++) { */
/* 	int index = iens * ens_stride + iobs * obs_stride; */
/* 	S[index] -= meanS[iobs]; */
/*       } */
    
/*     free(S1); */
/*     free(S2); */
/*   } */
  
  return S;
}

