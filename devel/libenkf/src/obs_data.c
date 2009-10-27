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

 2. The functions meas_matrix_alloc_stats() is called to calculate
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
#include <matrix.h>


/* Small type holding one observation  - could be enhanced for non-diagonal covariance observation errors.*/

struct obs_data_node_struct {
  double        value;
  double        std; 
  const char   *keyword;  /* 
                             This keyword is ONLY used for pretty-printing. Observe that the
                             storage for this keyword is NOT owned by the obs_data_node
                             structure, but rather by the type specific field_obs || gen_obs ||
                             summary_obs structure.
                          */
  bool          active;
};


static obs_data_node_type * obs_data_node_alloc( double value , double std , const char * keyword) {
  obs_data_node_type * node = util_malloc( sizeof * node , __func__);
  node->value   = value;
  node->std     = std;
  node->keyword = keyword;
  node->active  = true;
  return node;
}


static void obs_data_node_free( obs_data_node_type * node ) {
  free( node );
}
  

const char * obs_data_node_get_keyword( const obs_data_node_type * node ) {
  return node->keyword;
}


double obs_data_node_get_std( const obs_data_node_type * node ) {
  return node->std;
}

double obs_data_node_get_value( const obs_data_node_type * node ) {
  return node->value;
}


bool obs_data_node_active( const obs_data_node_type * node ) {
  return node->active;
}

/*****************************************************************/


struct obs_data_struct {
  int       total_size;     /* The number of measurements which have been added with obs_data_add. */
  int       active_size;    /* The number of the measurements which are active (some might have been deactivated). */ 
  int       alloc_size;     /* The size of the value and std buffers. */ 
  int       target_size;    /* We aim for this size of the buffers - if alloc_size is currently larger, it will shrink on reset. */ 

  obs_data_node_type ** data;
}; 


static void obs_data_realloc_data(obs_data_type * obs_data, int new_alloc_size) {
  int old_alloc_size      = obs_data->alloc_size;
  obs_data->alloc_size    = new_alloc_size;
  obs_data->data          = util_realloc(obs_data->data , new_alloc_size * sizeof * obs_data->data   	  , __func__);
  {
    int i;
    for (i= old_alloc_size; i < new_alloc_size; i++)
      obs_data->data[i] = NULL;
  }
}

int obs_data_get_nrobs(const obs_data_type * obs_data) { return obs_data->total_size; }



obs_data_type * obs_data_alloc() {
  obs_data_type * obs_data = malloc(sizeof * obs_data);
  obs_data->alloc_size    = 0;
  obs_data->target_size   = 0;
  obs_data->data          = NULL;
  

  obs_data_realloc_data(obs_data , obs_data->target_size);
  obs_data_reset(obs_data);
  return obs_data;
}


void obs_data_iget_value_std(const obs_data_type * obs_data , int index , double * value ,  double * std) {
  obs_data_node_type * node = obs_data->data[index];
  *value = node->value;
  *std   = node->std;
}


obs_data_node_type * obs_data_iget_node( const obs_data_type * obs_data , int index ) {
  return obs_data->data[index];
}


void obs_data_reset(obs_data_type * obs_data) { 
  obs_data->total_size  = 0; 
  obs_data->active_size = 0; 
  if (obs_data->alloc_size > obs_data->target_size)
    obs_data_realloc_data(obs_data , obs_data->target_size);

}


void obs_data_add(obs_data_type * obs_data, double value, double std , const char *kw) {
  if (obs_data->total_size == obs_data->alloc_size)
    obs_data_realloc_data(obs_data , 2*obs_data->alloc_size + 2);
  {
    int index = obs_data->total_size;
    obs_data_node_type * node 	 = obs_data_node_alloc( value , std , kw );
    obs_data->data[index]     	 = node;
  }
  obs_data->total_size++;
  obs_data->active_size++;
}



void obs_data_free(obs_data_type * obs_data) {
  int i;
  for (i=0; i < obs_data->alloc_size; i++) {
    if (obs_data->data[i] != NULL)
      obs_data_node_free( obs_data->data[i] );
  }
  free(obs_data->data);
  free(obs_data);
}



matrix_type * obs_data_allocE__(const obs_data_type * obs_data , int ens_size) {
const int nrobs_active = obs_data->active_size;
  const int nrobs_total  = obs_data->total_size;
  double *pert_mean , *pert_var;
  matrix_type * E;
  int iens, iobs_active;
  
  E         = matrix_alloc( nrobs_active , ens_size);

  pert_mean = util_malloc(nrobs_active            * sizeof * pert_mean , __func__);
  pert_var  = util_malloc(nrobs_active            * sizeof * pert_var  , __func__);
  {
    double * tmp = util_malloc( nrobs_active * ens_size * sizeof * tmp , __func__);
    int i,j;
    int k = 0;
    enkf_util_rand_stdnormal_vector(nrobs_active * ens_size , tmp);
    for (j=0; j < ens_size; j++)
      for (i=0; i < nrobs_active; i++) {
	matrix_iset( E , i , j , tmp[k]);
	k++;
      }
    free(tmp);
  }
  
  for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
    pert_mean[iobs_active] = 0;
    pert_var[iobs_active]  = 0;
  }
  
  for (iens = 0; iens < ens_size; iens++) 
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) 
      pert_mean[iobs_active] += matrix_iget(E , iobs_active , iens);


  for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) 
    pert_mean[iobs_active] /= ens_size;

  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
      double tmp;
      matrix_iadd(E , iobs_active , iens , -pert_mean[iobs_active]);
      tmp = matrix_iget(E , iobs_active , iens);
      pert_var[iobs_active] += tmp * tmp;
    }
  }
  
  {
    int iobs_total;
    for (iens = 0; iens < ens_size; iens++) {
      int iobs_active = 0;
      for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
	const obs_data_node_type * node = obs_data->data[iobs_total];
	if (node->active) {
	  obs_data_node_type * node = obs_data->data[iobs_total];
	  matrix_imul(E , iobs_active , iens , node->std * sqrt(ens_size / pert_var[iobs_active]));
	  iobs_active++;
	}
      }
    }
  }
  

  free(pert_mean);
  free(pert_var);
  return E;
}







matrix_type * obs_data_allocD__(const obs_data_type * obs_data , const matrix_type * E  , const matrix_type * S) {
  const int nrobs_active = obs_data->active_size;
  const int nrobs_total  = obs_data->total_size;
  int ens_size           = matrix_get_columns( S );              
  int iobs_active, iobs_total;
  int iens;
  
  matrix_type * D = matrix_alloc( nrobs_active , ens_size);
  for  (iens = 0; iens < ens_size; iens++) {
    iobs_active = 0;
    for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
      obs_data_node_type * node = obs_data->data[iobs_total];
      if (node->active) {
	matrix_iset(D , iobs_active , iens , matrix_iget(E , iobs_active , iens)  - matrix_iget(S, iobs_active , iens) + node->value);
	iobs_active++;
      }
    }
  }

  return D;
}







/**
  This function deactivates obsveration iobs, and decrements the total
  number of active observations.
*/
void obs_data_deactivate_obs(obs_data_type * obs_data , int iobs,const char * msg) {
  obs_data_node_type * node = obs_data->data[iobs];
  if (node->active) {
    node->active = false;
    obs_data->active_size--;
    printf("Deactivating:%s : %s \n",node->keyword , msg);
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
	De activated because the ensemble has to small variation for
	this particular measurement.
      */
      obs_data_deactivate_obs(obs_data , iobs , "No ensemble variation");
    else {
      obs_data_node_type * node = obs_data->data[iobs];
      if (fabs( innov[iobs] ) > alpha * (ens_std[iobs] + node->std))
	/* OK - this is an outlier ... */
	obs_data_deactivate_obs(obs_data , iobs , "No overlap");
    }
  }
  *nrobs_active = obs_data->active_size;
  {
    *active_obs = NULL; /* This will break on old code. */
  }
}


matrix_type * obs_data_allocR__(obs_data_type * obs_data) {
  const int nrobs_total  = obs_data->total_size;
  const int nrobs_active = obs_data->active_size;
  matrix_type * R;
  int iobs_active;

  R = matrix_alloc( nrobs_active , nrobs_active );
  {
    int iobs_total;
    iobs_active = 0;

    for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
      obs_data_node_type * node = obs_data->data[iobs_total];
      if (node->active) {
	double std = node->std;
	matrix_iset(R , iobs_active , iobs_active , std * std);
	iobs_active++;
      }
    }
  }
  
  return R;
}





double * obs_data_alloc_innov__(const obs_data_type * obs_data , const meas_matrix_type * meas_matrix) {
  double *innov;
  int nrobs_total  = obs_data->total_size;
  int nrobs_active = obs_data->active_size;
  int iobs_total;
  int iobs_active = 0;

  innov = util_malloc(nrobs_active * sizeof * innov , __func__);
  for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
    obs_data_node_type * node = obs_data->data[iobs_total];
    if (node->active) {
      innov[iobs_active] = node->value - meas_matrix_iget_ens_mean( meas_matrix , iobs_total );
      iobs_active++;
    }
  }

  return innov;
}




void obs_data_scale__(const obs_data_type * obs_data , matrix_type *S , matrix_type *E , matrix_type *D , matrix_type *R , double *innov) {
  const int nrobs_total  = obs_data->total_size;
  const int nrobs_active = obs_data->active_size;
  const int ens_size     = matrix_get_columns( S );
  double * scale_factor  = util_malloc(nrobs_active * sizeof * scale_factor , __func__);
  int iens, iobs_active;
  
  {
    int iobs_total;
    iobs_active = 0;
    for (iobs_total = 0; iobs_total < nrobs_total; iobs_total++) {
      obs_data_node_type * node = obs_data->data[iobs_total];
      if (node->active) {
	scale_factor[iobs_active] = 1.0 / node->std;
	iobs_active++;
      }
    }
  }
  
  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) {
      matrix_imul(S , iobs_active , iens , scale_factor[iobs_active]);

      if (D != NULL)
        matrix_imul(D , iobs_active , iens , scale_factor[iobs_active]);

      if (E != NULL)
	matrix_imul(E , iobs_active , iens , scale_factor[iobs_active]);
    }
  }
  
  if (innov != NULL)
    for (iobs_active = 0; iobs_active < nrobs_active; iobs_active++) 
      innov[iobs_active] *= scale_factor[iobs_active];
  
  {
    for (int i=0; i < nrobs_active; i++)
      for (int j=0; j < nrobs_active; j++)
	matrix_imul(R , i , j , scale_factor[i] * scale_factor[j]);
  }
  free(scale_factor);
}



void obs_data_fprintf(const obs_data_type * obs_data , FILE * stream, const double * meanS , const double * stdS) {
  int iobs;
  fprintf(stream , "/-------------------------------------------------------------------------------------------|---------------------------------\\\n");
  fprintf(stream , "|                                     Observed history                                      |         Simulated data          |\n");  
  fprintf(stream , "|-------------------------------------------------------------------------------------------|---------------------------------|\n");
  for (iobs = 0; iobs < obs_data->total_size; iobs++) {
    obs_data_node_type * node = obs_data->data[iobs];
    fprintf(stream , "| %-3d : %-36s    %12.3f +/-  %12.3f ",iobs + 1 , node->keyword , node->value , node->std);
    if (node->active)
      fprintf(stream , "   Active    |");
    else
      fprintf(stream , "   Inactive  |");

    if (meanS != NULL) 
      fprintf(stream,"   %12.3f +/- %12.3f |\n",meanS[iobs] , stdS[iobs]);
    else
      fprintf(stream , " .... |\n");
    
  }
  fprintf(stream , "\\-------------------------------------------------------------------------------------|---------------------------------/\n");
}


int obs_data_get_active_size( const obs_data_type  * obs_data ) {
  return obs_data->active_size;
}



