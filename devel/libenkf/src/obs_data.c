/**
See the file README.obs for ducumentation of the varios datatypes
involved with observations/measurement/+++.


The file contains two different variables holding the number of
observations, nrobs_total and nrobs_active. The first holds the total
number of observations at this timestep, and the second holds the
number of active measurements at this timestep; the inactive
measurements have been deactivated the obs_data_deactivate_outliers()
function.

The flow is as follows:

 1. All the observations have been collected in an obs_data instance,
    and all the corresponding measurements of the state have been
    collected in a meas_matrix instance - we are ready for analysis.

 2. The functions meas_matrix_allocS_stats() is called to calculate
    the ensemble mean and std of all the measurements.

 3. The function obs_data_deactivate_outliers() is called to compare
    the ensemble mean and std with the observations, in the case of
    outliers the number obs_active flag of the obs_data instance is
    set to false.

 4. The remaining functions (and matrices) now refer to the number of
    active observations, however the "raw" observations found in the
    obs_data instance are in a vector with nrobs_total observations;
    i.e. we must handle two indices and two total lengths. A bit
    messy.


Variables of size nrobs_total:
------------------------------
 o obs->value / obs->std / obs->obs_active
 o meanS , innov, stdS


variables of size nrobs_active:
-------------------------------
Matrices: S, D, E and various internal variables.
*/




#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <enkf_util.h>
#include <obs_data.h>
#include <util.h>
#include <meas_matrix.h>

struct obs_data_struct {
  int       active_size;
  int       total_size;
  int       alloc_size;
  double   *value;
  double   *std;
  char    **keyword;
  bool     *obs_active;
}; 


static void obs_data_realloc_data(obs_data_type * obs_data, int new_alloc_size) {
  int old_alloc_size      = obs_data->alloc_size;
  obs_data->alloc_size    = new_alloc_size;
  obs_data->value         = enkf_util_realloc(obs_data->value         , new_alloc_size * sizeof * obs_data->value   	  , __func__);
  obs_data->std           = enkf_util_realloc(obs_data->std           , new_alloc_size * sizeof * obs_data->std     	  , __func__);
  obs_data->keyword       = enkf_util_realloc(obs_data->keyword       , new_alloc_size * sizeof * obs_data->keyword 	  , __func__);
  obs_data->obs_active    = enkf_util_realloc(obs_data->obs_active    , new_alloc_size * sizeof * obs_data->obs_active    , __func__);
  {
    int i;
    for (i= old_alloc_size; i < new_alloc_size; i++)
      obs_data->keyword[i] = NULL;
  }
}


int obs_data_get_nrobs(const obs_data_type * obs_data) { return obs_data->total_size; }

obs_data_type * obs_data_alloc() {
  obs_data_type * obs_data = malloc(sizeof * obs_data);
  obs_data->total_size    = 0;
  obs_data->active_size   = 0;
  obs_data->value      	  = NULL;
  obs_data->std        	  = NULL;
  obs_data->keyword    	  = NULL;
  obs_data->obs_active    = NULL;

  obs_data->alloc_size = 0;
  obs_data_realloc_data(obs_data , 10);
  return obs_data;
}



void obs_data_reset(obs_data_type * obs_data) { 
  obs_data->total_size = 0; 
  obs_data->active_size = 0; 
}


void obs_data_add(obs_data_type * obs_data, double value, double std , const char *kw) {
  if (obs_data->total_size == obs_data->alloc_size)
    obs_data_realloc_data(obs_data , 2*obs_data->alloc_size + 2);
  {
    int index = obs_data->total_size;
    obs_data->value[index]   	  = value;
    obs_data->std[index]     	  = std;
    obs_data->keyword[index]      = util_realloc_string_copy(obs_data->keyword[index] , kw);
    obs_data->obs_active[index]   = true;
  }
  obs_data->total_size++;
  obs_data->active_size++;
}



void obs_data_free(obs_data_type * obs_data) {
  free(obs_data->value);
  free(obs_data->std);
  free(obs_data->obs_active);
  util_free_string_list(obs_data->keyword , obs_data->total_size);
  free(obs_data);
}





static double * obs_data_allocE(const obs_data_type * obs_data , int ens_size, int ens_stride, int obs_stride) {
  const int nrobs_active = obs_data->active_size;
  const int nrobs_total  = obs_data->total_size;
  double *pert_mean , *pert_var;
  double *E;
  int iens, iobs_active;
  
  E         = util_malloc(nrobs_active * ens_size * sizeof * E         , __func__);

  pert_mean = util_malloc(nrobs_active            * sizeof * pert_mean , __func__);
  pert_var  = util_malloc(nrobs_active            * sizeof * pert_var  , __func__);
  enkf_util_rand_stdnormal_vector(nrobs_active * ens_size , E);
  
  for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
    pert_mean[iobs_active] = 0;
    pert_var[iobs_active]  = 0;
  }
  
  for (iens = 0; iens < ens_size; iens++) {
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
      int index = iens * ens_stride + iobs_active * obs_stride;
      pert_mean[iobs_active] += E[index];
    }
  }


  for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) 
    pert_mean[iobs_active] /= ens_size;

  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
      int index = iens * ens_stride + iobs_active * obs_stride;
      E[index] -= pert_mean[iobs_active];
      pert_var[iobs_active] += E[index] * E[index];
    }
  }

  {
    int iobs_total;
    for (iens = 0; iens < ens_size; iens++) {
      int iobs_active = 0;
      for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
	if (obs_data->obs_active[iobs_total]) {
	  int index = iens * ens_stride + iobs_active * obs_stride;
	  E[index] *= obs_data->std[iobs_total] * sqrt(ens_size / pert_var[iobs_active]);
	  iobs_active++;
	}
      }
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
double * obs_data_allocD(const obs_data_type * obs_data , int ens_size, int ens_stride , int obs_stride , const double * S , const double * meanS , bool returnE , double **_E) {
  const int nrobs_active = obs_data->active_size;
  const int nrobs_total  = obs_data->total_size;
  int iobs_active, iobs_total;
  int iens;
  double *D = NULL;
  double *E = NULL;


  E  = obs_data_allocE(obs_data , ens_size , ens_stride , obs_stride);
  D  = util_malloc(nrobs_active * ens_size * sizeof * D , __func__);

  for  (iens = 0; iens < ens_size; iens++) {
    iobs_active = 0;
    for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
      if (obs_data->obs_active[iobs_total]) {
	int index = iens * ens_stride + iobs_active * obs_stride;
	D[index]  = obs_data->value[iobs_total] + E[index] - S[index] - meanS[iobs_active];
	iobs_active++;
      }
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



/**
  Observe that this function is called two times; first with all
  measurements active, and then with (possibly) some measurements
  deactivated.

  The input vector meanS should contain the same number of elements
  as is currently active.
*/

double * obs_data_alloc_innov(const obs_data_type * obs_data , const double *meanS) {
  double *innov;
  int nrobs_total  = obs_data->total_size;
  int nrobs_active = obs_data->active_size;
  int iobs_total;
  int iobs_active = 0;

  innov = util_malloc(nrobs_active * sizeof * innov , __func__);
  for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
    if (obs_data->obs_active[iobs_total]) {
      innov[iobs_active] = obs_data->value[iobs_total] - meanS[iobs_active];
      iobs_active++;
    }
  }

  return innov;
}



/**
  This function deactivates obsveration iobs, and decrements the total
  number of active observations.
*/
static void obs_data_deactivate_obs(obs_data_type * obs_data , int iobs,const char * msg) {
  if (obs_data->obs_active[iobs]) {
    obs_data->obs_active[iobs] = false;
    obs_data->active_size--;
    printf("Deactivating:%s : %s \n",obs_data->keyword[iobs] , msg);
  }
}


/**
This code deactivates outliers. This is done based on two different
principles:

 o If the ensemble variation of a particular measurement is zero (for
   instance all members achieved the target rate). This decision is
   based solely on the measurements of the ensemble; the observations
   are not taken into account.

 o If the overlap between ensemble prediction and observed data is too
   small.

Ideally these two requirements should be combined into one common
statistic?
*/




void obs_data_deactivate_outliers(obs_data_type * obs_data , const double * innov , const double *ens_std , double std_cutoff , double alpha , int * nrobs_active , bool **active_obs) {
  int nrobs = obs_data->total_size;
  int iobs;
  for (iobs = 0; iobs < nrobs; iobs++) {
    if (ens_std[iobs] < std_cutoff)
      /*
	De activated because the ensemble has to little variation for
	this particular measurement.
      */
      {
	printf("stdS : %g \n",ens_std[iobs]);	
	obs_data_deactivate_obs(obs_data , iobs , "No ensemble variation");
      }
    else {
      if (fabs( innov[iobs] ) > alpha * (ens_std[iobs] + obs_data->std[iobs])) 
	/* OK - this is an outlier ... */
	obs_data_deactivate_obs(obs_data , iobs , "No overlap");
    }
  }
  *nrobs_active = obs_data->active_size;
  *active_obs   = obs_data->obs_active;
}


double * obs_data_allocR(obs_data_type * obs_data) {
  const int nrobs_total  = obs_data->total_size;
  const int nrobs_active = obs_data->active_size;
  double *R;
  int iobs_active;

  R = util_malloc( nrobs_active * nrobs_active * sizeof * R , __func__);
  for (iobs_active = 0; iobs_active < nrobs_active * nrobs_active; iobs_active++)
    R[iobs_active] = 0;
  
  {
    int iobs_total;
    iobs_active = 0;

    for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
      if (obs_data->obs_active[iobs_total]) {
	double std = obs_data->std[iobs_total];
	R[iobs_active * (nrobs_active + 1)] = std * std;
	iobs_active++;
      }
    }
  }
  
  return R;
}



void obs_data_scale(const obs_data_type * obs_data , int ens_size, int ens_stride , int obs_stride , double *S , double *E , double *D , double *R , double *innov) {
  const int nrobs_total  = obs_data->total_size;
  const int nrobs_active = obs_data->active_size;
  double * scale_factor  = util_malloc(nrobs_active * sizeof * scale_factor , __func__);
  int iens, iobs_total , iobs_active;
  
  {
    int iobs_total;
    iobs_active = 0;
    for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
      if (obs_data->obs_active[iobs_total]) {
	scale_factor[iobs_active] = 1.0 / obs_data->std[iobs_total];
	iobs_active++;
      }
    }
  }
  
  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
      int index                = iens * ens_stride + iobs_active * obs_stride;
      S[index]  	      *= scale_factor[iobs_active];
      D[index]  	      *= scale_factor[iobs_active];
      if (E != NULL) E[index] *= scale_factor[iobs_active];
    }
  }
  
  for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) 
    innov[iobs_active] *= scale_factor[iobs_active];
  
  {
    int i,j;
    for (i=0; i < nrobs_active; i++)
      for (j=0; j < nrobs_active; j++)
	R[i*nrobs_active + j] *= (scale_factor[i] * scale_factor[j]);
  }
  free(scale_factor);
}



void obs_data_fprintf(const obs_data_type * obs_data , FILE * stream, const double * meanS , const double * stdS) {
  int iobs;
  fprintf(stream , "--------------------------------------------------------------------\n");
  for (iobs = 0; iobs < obs_data->total_size; iobs++) {
    fprintf(stream , "%-3d : %-16s    %12.3f +/-  %12.3f ",iobs + 1 , obs_data->keyword[iobs] , obs_data->value[iobs] , obs_data->std[iobs]);
    if (obs_data->obs_active[iobs])
      fprintf(stream , "   Active    |");
    else
      fprintf(stream , "   Inactive  |");

    if (meanS != NULL) 
      fprintf(stream,"   %12.3f +/- %12.3f \n",meanS[iobs] , stdS[iobs]);
    else
      fprintf(stream , "\n");
    
  }
  fprintf(stream , "--------------------------------------------------------------------\n");
}
