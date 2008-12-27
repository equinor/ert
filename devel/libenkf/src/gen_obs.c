/**
   See the overview documentation of the observation system in enkf_obs.c
*/
#include <stdlib.h>
#include <enkf_util.h>
#include <enkf_types.h>
#include <enkf_macros.h>
#include <util.h>
#include <gen_obs.h>
#include <meas_vector.h>
#include <obs_data.h>
#include <gen_data.h>
#include <gen_obs.h>
#include <gen_common.h>
#include <gen_obs_active.h>

/**
   This file implemenets a structure for general observations. A
   general observation is just a vector of numbers - where EnKF has no
   understanding whatsover of the type of these data.

   When residuals are calculated a gen_obs instance should (quite
   obviously) be compared with a gen_data instance. The actual data is
   supposed to be found in a file.
*/



#define GEN_OBS_TYPE_ID 77619

struct gen_obs_struct {
  int                     __type_id;
  int                     data_size;       /* This is the total size of the observation vector. */ 
  int                     active_size;     /* This is the number of active indices. */
  int                   * active_list;     /* This is a a list of indices which are active - can be NULL if all is active. */
  int                   * data_index_list; /* The indexes which are observed in the corresponding gen_data instance.*/
  double                * __obs_buffer;    /* This is the actual storage variable. obs_data and obs_std just point into this vector. */
  double                * obs_data;        /* The observed data. */
  double                * obs_std;         /* The observed standard deviation. */ 
  char                  * obs_file;        /* The file holding the observation. */ 
  gen_data_format_type    obs_format;      /* The format, i.e. ASCII, binary_double or binary_float, of the observation file. */
};

/******************************************************************/


void gen_obs_free(gen_obs_type * gen_obs) {
  util_safe_free(gen_obs->__obs_buffer);
  util_safe_free(gen_obs->obs_file);
  util_safe_free(gen_obs->data_index_list);
  free(gen_obs);
}




gen_obs_type * gen_obs_alloc(const char * obs_file) {
  gen_obs_type * obs = util_malloc(sizeof * obs , __func__);
  
  obs->__type_id       = GEN_OBS_TYPE_ID;
  obs->active_size     = 0;
  obs->active_list     = NULL;
  obs->__obs_buffer    = NULL;
  obs->data_index_list = NULL;
  obs->obs_file        = util_alloc_string_copy( obs_file );
  obs->obs_format      = ASCII;
  return obs;
}



void gen_obs_measure(const gen_obs_type * gen_obs , const gen_data_type * gen_data , meas_vector_type * meas_vector) {
  
  int iobs;
  if (gen_obs->data_index_list == NULL)
    util_abort("%s: somehow no one allocated the gen_obs->data_index_list popinter - hmmm \n",__func__);
  
  if (gen_obs->active_size == gen_obs->data_size) {
    for (iobs = 0; iobs < gen_obs->data_size; iobs++)
      meas_vector_add( meas_vector , gen_data_iget_double( gen_data , gen_obs->data_index_list[iobs] ));
  } else 
    for (iobs = 0; iobs < gen_obs->active_size; iobs++)
      meas_vector_add( meas_vector , gen_data_iget_double( gen_data , gen_obs->active_list[gen_obs->data_index_list[iobs]] ));
  
}




/**
   This functions loads observations from file; collects the
   observations which are active, and free the observations again.

   The file with observations should be a long vector of 2N elements,
   where the first N elements are data values, and the last N values
   are the corresponding standard deviations.

   The file is loaded with the gen_common_fload_alloc() function, and
   can be in formatted ASCII or binary_float / binary_double. Observe
   that there is *NO* header information in this file.
*/

void gen_obs_get_observations(gen_obs_type * gen_obs , int report_step, obs_data_type * obs_data) {
  int iobs;
  const char * kw = "GEN_OBS";
  ecl_type_enum load_type;
  
  gen_obs->__obs_buffer = gen_common_fload_alloc(gen_obs->obs_file , gen_obs->obs_format , ecl_double_type , &load_type , &gen_obs->data_size);
  if (load_type == ecl_float_type) {
    double * double_data = util_malloc(gen_obs->data_size * sizeof * double_data , __func__);
    util_float_to_double(double_data , (const float *) gen_obs->__obs_buffer , gen_obs->data_size);
    free(gen_obs->__obs_buffer);
    gen_obs->__obs_buffer = double_data;
  }

  gen_obs->data_size /= 2; /* Originally contains BOTH data and std. */
  gen_obs->obs_data   =  gen_obs->__obs_buffer;
  gen_obs->obs_std    = &gen_obs->__obs_buffer[gen_obs->data_size];
  
  if (gen_obs->active_size == gen_obs->data_size) {
    for (iobs = 0; iobs < gen_obs->data_size; iobs++)
      obs_data_add( obs_data , gen_obs->obs_data[iobs] , gen_obs->obs_std[iobs] , kw);
  } else 
    for (iobs = 0; iobs < gen_obs->active_size; iobs++)
      obs_data_add( obs_data , gen_obs->obs_data[gen_obs->active_list[iobs]] , gen_obs->obs_std[gen_obs->active_list[iobs]] , kw);

  gen_obs->__obs_buffer = util_safe_free(gen_obs->__obs_buffer);


  gen_obs->data_index_list = util_malloc( sizeof * gen_obs->data_index_list * gen_obs->active_size, __func__);

  /* Defaulting to :

     1. Conformal vectors for observations and data.
     2. Observing the whole shit.
     
  */
  for (iobs = 0; iobs < gen_obs->data_size; iobs++)
    gen_obs->data_index_list[iobs] = iobs;
}




void gen_obs_activate(gen_obs_type * obs , active_mode_type active_mode , void * __active) {
  gen_obs_active_type * active = gen_obs_active_safe_cast(__active);
  
  util_safe_free(obs->active_list);
  if (active_mode == all_active)
    obs->active_size = obs->data_size;
  else if (active_mode == inactive)
    obs->active_size = 0;
  else if (active_mode == partly_active) {
    obs->active_size = gen_obs_active_get_active_size( active );
    obs->active_list = gen_obs_active_alloc_list_copy( active );
  } else 
    util_abort("%s: internal error - active_mode:%d completely invalid \n",__func__ , active_mode);
}


void gen_obs_user_get(const gen_obs_type * gen_obs , const char * index_key , double * value , double * std , bool * valid) {
  *valid = true;
  *value = 1.0;
  *std   = 1.0;
}


  
/*****************************************************************/
SAFE_CAST(gen_obs , GEN_OBS_TYPE_ID)
IS_INSTANCE(gen_obs , GEN_OBS_TYPE_ID)
VOID_OBS_ACTIVATE(gen_obs)
VOID_FREE(gen_obs)
VOID_GET_OBS(gen_obs)
VOID_MEASURE(gen_obs , gen_data)
VOID_USER_GET_OBS(gen_obs)
