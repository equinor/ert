#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <enkf_util.h>
#include <field_obs.h> 
#include <obs_data.h>
#include <meas_op.h>
#include <meas_vector.h>
#include <hash.h>
#include <list.h>
#include <field_config.h>
#include <field.h>



struct field_obs_struct {
  const field_config_type * config;
  int    	            size;
  int                     * i , * j , * k;
  double               	  *  obs_std;
  double                  *  obs_value;
  char                    *  obs_name;
};



static field_obs_type * __field_obs_alloc(const field_config_type * config , const char * obs_name , int size , const int *i ,const int *j , const int *k, const double * obs_value , const double * obs_std) {
  
  field_obs_type * field_obs = malloc(sizeof * field_obs);
  field_obs->size      	     = size;
  field_obs->config          = config;
  field_obs->obs_name        = util_alloc_string_copy(obs_name);
  field_obs->i               = util_alloc_copy(i 	  , size * sizeof * field_obs->i 	 , __func__);
  field_obs->j               = util_alloc_copy(j 	  , size * sizeof * field_obs->j 	 , __func__);
  field_obs->k               = util_alloc_copy(k 	  , size * sizeof * field_obs->k 	 , __func__);
  field_obs->obs_value       = util_alloc_copy(obs_value , size * sizeof * field_obs->obs_value , __func__);
  field_obs->obs_std         = util_alloc_copy(obs_std   , size * sizeof * field_obs->obs_value , __func__);
  
  return field_obs;
}




field_obs_type * field_obs_alloc(const field_config_type * config , const char * obs_name , int size , const int *i , const int *j , const int * k, const double * obs_value , const double * obs_std ) {
  return __field_obs_alloc(config , obs_name , size , i , j , k, obs_value , obs_std);
}



field_obs_type * field_obs_fscanf_alloc(const char * filename , const field_config_type * config) {
  fprintf(stderr,"%s: not implemented - aborting \n",__func__);
  return NULL;
}






void field_obs_get_observations(const field_obs_type * field_obs , int report_step, obs_data_type * obs_data) {
  char  * obs_name = NULL;
  int i;
  for (i=0; i < field_obs->size; i++) {
    obs_name = util_realloc(obs_name , strlen(field_obs->obs_name) + 30 , __func__);
    sprintf(obs_name , "%s i=%3d, j=%3d, k=%3d" , field_obs->obs_name , field_obs->i[i] , field_obs->j[i] , field_obs->k[i]);
    obs_data_add(obs_data , field_obs->obs_value[i] , field_obs->obs_std[i] , obs_name);
  }
  if (obs_name != NULL) free(obs_name);
}




void field_obs_measure(const field_obs_type * field_obs , const field_type * field_state , meas_vector_type * meas_vector) {
  int i;
  double value;
  /* Should check type of field */
  for (i=0; i < field_obs->size; i++) {
    field_ijk_get(field_state , field_obs->i[i] , field_obs->j[i] , field_obs->k[i] , &value);
    meas_vector_add(meas_vector , value);
  }
}




void field_obs_free(field_obs_type * field_obs) {
  free(field_obs->i);
  free(field_obs->j);
  free(field_obs->k);
  free(field_obs->obs_value);
  free(field_obs->obs_std);
  free(field_obs->obs_name);
  free(field_obs);
}




VOID_FREE(field_obs)
VOID_GET_OBS(field_obs)
VOID_MEASURE(field)
