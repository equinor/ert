#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <enkf_util.h>
#include <obs_data.h>
#include <util.h>
#include <meas_matrix.h>

struct obs_data_struct {
  int       size;
  int       alloc_size;
  double   *value;
  double   *std;
  double   *std_inflation;
  char    **keyword;
}; 


static void obs_data_realloc_data(obs_data_type * obs_data, int new_alloc_size) {
  int old_alloc_size      = obs_data->alloc_size;
  obs_data->alloc_size    = new_alloc_size;
  obs_data->value         = enkf_util_realloc(obs_data->value         , new_alloc_size * sizeof * obs_data->value   	  , __func__);
  obs_data->std           = enkf_util_realloc(obs_data->std           , new_alloc_size * sizeof * obs_data->std     	  , __func__);
  obs_data->keyword       = enkf_util_realloc(obs_data->keyword       , new_alloc_size * sizeof * obs_data->keyword 	  , __func__);
  obs_data->std_inflation = enkf_util_realloc(obs_data->std_inflation , new_alloc_size * sizeof * obs_data->std_inflation , __func__);
  {
    int i;
    for (i= old_alloc_size; i < new_alloc_size; i++)
      obs_data->keyword[i] = NULL;
  }
}


int obs_data_get_nrobs(const obs_data_type * obs_data) { return obs_data->size; }

obs_data_type * obs_data_alloc() {
  obs_data_type * obs_data = malloc(sizeof * obs_data);
  obs_data->size       	  = 0;
  obs_data->value      	  = NULL;
  obs_data->std        	  = NULL;
  obs_data->keyword    	  = NULL;
  obs_data->std_inflation = NULL;

  obs_data->alloc_size = 0;
  obs_data_realloc_data(obs_data , 10);
  return obs_data;
}



void obs_data_reset(obs_data_type * obs_data) { 
  obs_data->size = 0; 
}


void obs_data_add(obs_data_type * obs_data, double value, double std , const char *kw) {
  if (obs_data->size == obs_data->alloc_size)
    obs_data_realloc_data(obs_data , 2*obs_data->alloc_size + 2);
  {
    int index = obs_data->size;
    obs_data->value[index]   	  = value;
    obs_data->std[index]     	  = std;
    obs_data->keyword[index]      = util_realloc_string_copy(obs_data->keyword[index] , kw);
  }
  obs_data->size++;
}



void obs_data_free(obs_data_type * obs_data) {
  free(obs_data->value);
  free(obs_data->std);
  free(obs_data->std_inflation);
  util_free_string_list(obs_data->keyword , obs_data->size);
  free(obs_data);
}



void obs_data_fprintf(const obs_data_type * obs_data , FILE *stream) {
  int i;
  for (i = 0; i < obs_data->size; i++)
    fprintf(stream , "%-3d : %-16s  %12.3f +/- %12.3f \n", i+1 , obs_data->keyword[i] , obs_data->value[i] , obs_data->std[i]);
}


static double * obs_data_allocE(const obs_data_type * obs_data , int ens_size, int ens_stride, int obs_stride) {
  double *pert_mean , *pert_var;
  double *E;
  int iens, iobs, nrobs;
  nrobs = obs_data->size;

  E         = util_malloc(nrobs * ens_size * sizeof * E        , __func__);
  pert_mean = util_malloc(nrobs            * sizeof * pert_mean , __func__);
  pert_var  = util_malloc(nrobs            * sizeof * pert_var  , __func__);
  enkf_util_rand_stdnormal_vector(nrobs * ens_size , E);
  
  for (iobs = 0; iobs < nrobs; iobs++) {
    pert_mean[iobs] = 0;
    pert_var[iobs]  = 0;
  }

  for (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      pert_mean[iobs] += E[index];
    }
  }
  for (iobs = 0; iobs < nrobs; iobs++) 
    pert_mean[iobs] /= ens_size;

  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      E[index] -= pert_mean[iobs];
      pert_var[iobs] += E[index] * E[index];
    }
  }
  
  for (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      E[index] *= obs_data->std[iobs] * sqrt(ens_size / pert_var[iobs]);
    }
  }
  
  free(pert_mean);
  free(pert_var);
  return E;
}




/**
  The D matrix 
  
  Observe that this code assumes that the ensemble mean has *not* 
  been subtracted from S. This is in contrast to the original
  fortran code, where it was assumed that ensemble mean was shifted 
  from S.
*/
double * obs_data_allocD(const obs_data_type * obs_data , int ens_size, int ens_stride , int obs_stride , const double * S , bool returnE , double **_E) {
  double *D , *E;
  int iens, iobs, nrobs;
  nrobs = obs_data->size;
  E 	= obs_data_allocE(obs_data , ens_size , ens_stride , obs_stride);
  D 	= util_malloc(nrobs * ens_size * sizeof * D , __func__);

  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      D[index] = obs_data->value[iobs] + E[index] - S[index];
    }
  }
  
  if (returnE)
    *_E = E;
  else {
    free(E);
    *_E = NULL;
  }

  return D;
}


/*
  Outliers are identified by an std_inflation > 1 - observe
  that the stored std is *NOT* inflated.
*/
 


double * obs_data_alloc_innov(const obs_data_type * obs_data , int ens_size , int ens_stride , int obs_stride ,  const double *S) {
  double *innov;
  double *S1;
  int iens, iobs, nrobs;
  nrobs = obs_data->size;
  innov = util_malloc(nrobs * sizeof *innov , __func__);
  S1    = util_malloc(nrobs * sizeof *S1 , __func__);

  for (iobs = 0; iobs < nrobs; iobs++) {
    innov[iobs] = obs_data->value[iobs];
    S1[iobs]    = 0.0;
  }

  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      S1[iobs] += S[index];
    }
  }

  for (iobs = 0; iobs < nrobs; iobs++) 
    innov[iobs] -= S1[iobs] / ens_size;
  
  free(S1);
  return innov;
}


double * obs_data_allocR(obs_data_type * obs_data , int ens_size , int ens_stride , int obs_stride , const double * innov , const double *S , double alpha) {
  const int nrobs = obs_data->size;
  double *ens_avg;
  double *ens_std;
  double *R;
  int iens, iobs;

  ens_std = util_malloc(nrobs * sizeof * ens_std , __func__);
  ens_avg = util_malloc(nrobs * sizeof * ens_avg , __func__);
  
  for (iobs = 0; iobs < nrobs; iobs++) {
    ens_std[iobs] = 0.0;
    ens_avg[iobs] = 0.0;
  }

  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      ens_std[iobs] += S[index] * S[index];
      ens_avg[iobs] += S[index];
    }
  }
  
  for (iobs = 0; iobs < nrobs; iobs++) {
    ens_avg[iobs] *= 1.0 / ens_size;
    ens_std[iobs] *= 1.0 / ens_size;
    ens_std[iobs]  = sqrt(ens_std[iobs] - ens_avg[iobs] * ens_avg[iobs]);
  }
  free(ens_avg);

  for (iobs = 0; iobs < nrobs; iobs++) {
    if (fabs( innov[iobs] ) > alpha * (ens_std[iobs] + obs_data->std[iobs])) {
      /* OK - this is an outlier ... */
      double new_obs_std            = (fabs(innov[iobs]) / alpha - ens_std[iobs]);
      obs_data->std_inflation[iobs] = new_obs_std / obs_data->std[iobs];
    } else 
      obs_data->std_inflation[iobs] = 1.0;
  }
  
  R = util_malloc( nrobs * nrobs * sizeof * R , __func__);
  for (iobs = 0; iobs < nrobs * nrobs; iobs++)
    R[iobs] = 0;
  
  for (iobs = 0; iobs < nrobs; iobs++) {
    double std = obs_data->std[iobs] * obs_data->std_inflation[iobs];
    R[iobs * (nrobs + 1)] = std * std;
  }

  free(ens_std);
  return R;
}



void obs_data_scale(const obs_data_type * obs_data , int ens_size, int ens_stride , int obs_stride , double *S , double *E , double *D , double *R , double *innov) {
  const int nrobs = obs_data->size;
  double * scale_factor = util_malloc(nrobs * sizeof * scale_factor , __func__);
  int iens, iobs;
  
  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index          = iens * ens_stride + iobs * obs_stride;
      scale_factor[iobs] = 1.0 / obs_data->std[iobs];
      
      S[index] *= scale_factor[iobs];
      if (E != NULL) E[index] *= scale_factor[iobs];
      D[index] *= scale_factor[iobs];
    }
  }
  for (iobs = 0; iobs < nrobs; iobs++) 
    innov[iobs] *= scale_factor[iobs];

  {
    int i,j;
    for (i=0; i < nrobs; i++)
      for (j=0; j < nrobs; j++)
	R[i*nrobs + j] *= scale_factor[i] * scale_factor[j];
  }
  free(scale_factor);
}
