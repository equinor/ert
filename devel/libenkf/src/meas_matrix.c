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
  int                 active_count; /* The number of active elements - initialized in meas_matrix_calculate_ens_stats(). */
  bool              * active;       /* */
  double            * ens_mean;     /* Mean over all ensemble members - explicitly calculated with meas_matrix_calculate_ens_stats() */
  double            * ens_std;      /* Std  over all ensemble members - explicitly calculated with meas_matrix_calculate_ens_stats() */
};



meas_matrix_type * meas_matrix_alloc(int ens_size) {
  meas_matrix_type * meas = malloc(sizeof * meas);
  if (ens_size <= 0) 
    util_abort("%s: ens_size must be > 0 - aborting \n",__func__);

  meas->ens_size     = ens_size;
  meas->meas_vectors = util_malloc(ens_size * sizeof * meas->meas_vectors , __func__);
  meas->active       = NULL;
  meas->ens_mean     = NULL;
  meas->ens_std      = NULL;
  {
    int i;
    for (i = 0; i < meas->ens_size; i++)
      meas->meas_vectors[i] = meas_vector_alloc();
  }
  return meas;
}


void meas_matrix_deactivate(meas_matrix_type * meas_matrix, int index) {
  meas_matrix->active[index] = false;
  meas_matrix->active_count -= 1;
}



void meas_matrix_free(meas_matrix_type * matrix) {
  int i;
  for (i=0; i < matrix->ens_size; i++)
    meas_vector_free(matrix->meas_vectors[i]);
  free(matrix->meas_vectors);
  util_safe_free( matrix->active );
  util_safe_free( matrix->ens_std );
  util_safe_free( matrix->ens_mean );
  free(matrix);
}



void meas_matrix_add(meas_matrix_type * matrix , int iens , double value) {
  meas_vector_add(matrix->meas_vectors[iens] , value);
}


void meas_matrix_reset(meas_matrix_type * matrix) {
  int iens;
  for (iens = 0; iens < matrix->ens_size; iens++) 
    meas_vector_reset(matrix->meas_vectors[iens]);
  matrix->active       = util_safe_free( matrix->active );
  matrix->ens_mean     = util_safe_free( matrix->ens_mean );
  matrix->ens_std      = util_safe_free( matrix->ens_std  );
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
    This function calculates the ensemble mean and standard deviation
    of the S matrix. The internal variables ens_mean and ens_std are
    allocated, and then filled with content in this function.

    This function M U S T be called before calling the
    meas_matrix_iget_ens_std()/meas_matrix_iget_ens_mean() functions.
*/

void meas_matrix_calculate_ens_stats(meas_matrix_type * matrix) {
  int nrobs            = meas_vector_get_size( matrix->meas_vectors[0] );
  matrix->active       = util_realloc( matrix->active   , sizeof * matrix->active   * nrobs , __func__);
  matrix->ens_std      = util_realloc( matrix->ens_std  , sizeof * matrix->ens_std  * nrobs , __func__);
  matrix->ens_mean     = util_realloc( matrix->ens_mean , sizeof * matrix->ens_mean * nrobs , __func__);
  {
    int iobs, iens;
    double *S1 = matrix->ens_mean;
    double *S2 = matrix->ens_std;
    
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
    
    matrix->ens_mean = S1;
    matrix->ens_std  = S2;
  
    /* Initializing all observations to active */
    matrix->active_count = 0;
    for (iobs = 0; iobs < nrobs; iobs++) {
      matrix->active[iobs] = true;
      matrix->active_count += 1;
    }
  }
}



/**
   Observe that the functions meas_matrix_calculate_ens_stats() 

      M U S T 
 
   be called before calling any of these three functions:
*/


double meas_matrix_iget_ens_mean(const meas_matrix_type * matrix , int index) {
  return matrix->ens_mean[index];
}


double meas_matrix_iget_ens_std(const meas_matrix_type * matrix , int index) {
  return matrix->ens_std[index];
}


void meas_matrix_iget_ens_mean_std( const meas_matrix_type * matrix , int index , double * mean , double * std) {
  *mean = matrix->ens_mean[index];
  *std  = matrix->ens_std[index];
}





 

matrix_type * meas_matrix_allocS__(const meas_matrix_type * matrix) {
  const int nrobs_total = meas_vector_get_nrobs(matrix->meas_vectors[0]);
  matrix_type * S;
  int iens , active_iobs;
  S     = matrix_alloc( matrix->active_count , matrix->ens_size);

  for (iens = 0; iens < matrix->ens_size; iens++) {
    const meas_vector_type * vector = matrix->meas_vectors[iens];
    if (nrobs_total != meas_vector_get_nrobs(vector)) 
      util_abort("%s: fatal internal error - not all measurement vectors equally long - aborting \n",__func__);
    
    {
      const double * meas_data = meas_vector_get_data_ref(vector);
      int total_iobs;
      active_iobs = 0;
      for (total_iobs = 0; total_iobs < nrobs_total; total_iobs++) {
	if (matrix->active[total_iobs]) {
	  matrix_iset(S , active_iobs , iens , meas_data[total_iobs]);
	  active_iobs++;
	}
      }
    }
  }

  /*
    The previous implementation substracted the ensemble mean here;
    the current implementation does not do that any longer.
  */
  return S;
}



int meas_matrix_get_nrobs( const meas_matrix_type * meas_matrix ) {
  return meas_vector_get_nrobs( meas_matrix->meas_vectors[0] );
}


int meas_matrix_get_ens_size( const meas_matrix_type * meas_matrix ) {
  return meas_matrix->ens_size;
}
