#include "ies_enkf_config.h"
#include "ies_enkf_data.h"

#include <ert/util/type_macros.hpp>
#include <ert/util/util.h>

//**********************************************
// IES "object" data definition
//**********************************************
/*
  The configuration data used by the ies_enkf module is contained in a
  ies_enkf_data_struct instance. The data type used for the ies_enkf
  module is quite simple; with only a few scalar variables, but there
  are essentially no limits to what you can pack into such a datatype.

  All the functions in the module have a void pointer as the first
  argument, this will immediately be casted to an ies_enkf_data_type
  instance, to get some type safety the UTIL_TYPE_ID system should be
  used.

  The data structure holding the data for your analysis module should
  be created and initialized by a constructor, which should be
  registered with the '.alloc' element of the analysis table; in the
  same manner the desctruction of this data should be handled by a
  destructor or free() function registered with the .freef field of
  the analysis table.
*/

#define IES_ENKF_DATA_TYPE_ID 6635831


struct ies_enkf_data_struct {
   UTIL_TYPE_ID_DECLARATION;
   int       iteration_nr;            // Keep track of the outer iteration loop
   double    ies_max_steplength;      // Maximum step length in Gauss Newton iteration
   double    ies_min_steplength;      // Minimum step length in Gauss Newton iteration
   double    ies_dec_steplength;      // Step length decline in Gauss Newton iteration
   double    gauss_newton_conv;       // NOT USED
   bool      ies_subspace;            // NOT USED
   int       max_gauss_newton_it;     // NOT USED
   int state_size;                    // Initial state_size used for checks in subsequent calls
   bool_vector_type * ens_mask0;      // Initial ensemble mask of active realizations
   bool_vector_type * ens_mask;       // Ensemble mask of active realizations
   bool_vector_type * obs_mask0;      // Initial observation mask for active measurements
   bool_vector_type * obs_mask;       // Current observation mask
   matrix_type * W;                   // Coefficient matrix used to compute Omega = I + W (I -11'/N)/sqrt(N-1)
   matrix_type * A0;                  // Prior ensemble used in Ei=A0 Omega_i
   matrix_type * E;                   // Prior ensemble of measurement perturations (should be the same for all iterations)
   bool      converged;               // GN has converged
   ies_enkf_config_type * config;     // This I don't understand but I assume I include data from the ies_enkf_config_type defined in ies_enkf_config.c
   FILE*     log_fp;                  // logfile id
};

UTIL_SAFE_CAST_FUNCTION( ies_enkf_data , IES_ENKF_DATA_TYPE_ID )
UTIL_SAFE_CAST_FUNCTION_CONST( ies_enkf_data , IES_ENKF_DATA_TYPE_ID )

void * ies_enkf_data_alloc( rng_type * rng) {
  ies_enkf_data_type * data = util_malloc( sizeof * data);
  UTIL_TYPE_ID_INIT( data , IES_ENKF_DATA_TYPE_ID );
  data->iteration_nr         = 0;
  data->ies_max_steplength   = 0.0;
  data->ies_min_steplength   = 0.0;
  data->ies_dec_steplength   = 0.0;
  data->gauss_newton_conv    = 0.0;
  data->max_gauss_newton_it  = 0;
  data->state_size           = 0;
  data->ens_mask0            = NULL;
  data->ens_mask             = NULL;
  data->obs_mask0            = NULL;
  data->obs_mask             = NULL;
  data->W                    = NULL;
  data->A0                   = NULL;
  data->E                    = NULL;
  data->converged            = false;
  data->config               = ies_enkf_config_alloc();
  data->log_fp               = NULL;
  return data;
}

void ies_enkf_data_free( void * arg ) {
  ies_enkf_data_type * data = ies_enkf_data_safe_cast( arg );
  ies_enkf_config_free( data->config );
  free( data );
}

void ies_enkf_data_set_iteration_nr( ies_enkf_data_type * data , int iteration_nr) {
  data->iteration_nr = iteration_nr;
}

int ies_enkf_data_inc_iteration_nr( ies_enkf_data_type * data) {
  data->iteration_nr++;
  return data->iteration_nr;
}

int ies_enkf_data_get_iteration_nr( const ies_enkf_data_type * data ) {
  return data->iteration_nr;
}


ies_enkf_config_type* ies_enkf_data_get_config(const ies_enkf_data_type * data) {
  return data->config;
}



void ies_enkf_data_update_ens_mask(ies_enkf_data_type * data, const bool_vector_type * ens_mask) {
  if (data->ens_mask)
    bool_vector_free(data->ens_mask);

  data->ens_mask = bool_vector_alloc_copy( ens_mask );
}


void ies_enkf_store_initial_obs_mask(ies_enkf_data_type * data, const bool_vector_type * obs_mask) {
  if (!data->obs_mask0)
    data->obs_mask0 = bool_vector_alloc_copy(obs_mask);
}

void ies_enkf_update_obs_mask(ies_enkf_data_type * data, const bool_vector_type * obs_mask) {
  if (data->obs_mask)
    bool_vector_free(data->obs_mask);

  data->obs_mask = bool_vector_alloc_copy(obs_mask);
}

int ies_enkf_data_get_obs_mask_size(const ies_enkf_data_type * data) {
  return bool_vector_size( data->obs_mask );
}

int ies_enkf_data_active_obs_count(const ies_enkf_data_type * data) {
  int nrobs_msk = ies_enkf_data_get_obs_mask_size( data );
  int nrobs = 0;
  for (int i = 0; i < nrobs_msk; i++) {
    if ( bool_vector_iget(data->obs_mask0,i) && bool_vector_iget(data->obs_mask,i) ){
      nrobs=nrobs+1;
    }
  }
  return nrobs;
}


int ies_enkf_data_get_ens_mask_size(const ies_enkf_data_type * data) {
  return bool_vector_size(data->ens_mask);
}


void ies_enkf_data_update_state_size(ies_enkf_data_type * data, int state_size) {
  if (data->state_size == 0)
    data->state_size = state_size;
}

FILE * ies_enkf_data_open_log(ies_enkf_data_type * data) {
  const char * ies_logfile = ies_enkf_config_get_ies_logfile( data->config );
  FILE * fp;
  if (data->iteration_nr == 1){
    fp = fopen(ies_logfile, "w");
  } else {
    fp = fopen(ies_logfile, "a");
  }
  data->log_fp = fp;
  return fp;
}

void ies_enkf_data_fclose_log(ies_enkf_data_type * data) {
  fflush(data->log_fp);
  fclose(data->log_fp);
}


void ies_enkf_data_store_initialE(ies_enkf_data_type * data, const matrix_type * E0) {
  if (!data->E){
    // We store the initial observation perturbations corresponding to data->obs_mask0.
    bool dbg = ies_enkf_config_get_ies_debug( data->config ) ;
    int m_nrobs      = util_int_min(matrix_get_rows( E0 )-1, 50);
    int m_ens_size   = util_int_min(matrix_get_columns( E0 )-1, 16);
    fprintf(data->log_fp,"Allocating and assigning data->E \n");
    data->E = matrix_alloc_copy(E0);
    if (dbg)
      matrix_pretty_fprint_submat(data->E,"data->E","%11.5f",data->log_fp,0,m_nrobs,0,m_ens_size);
  }
}

void ies_enkf_data_store_initialA(ies_enkf_data_type * data, const matrix_type * A) {
  if (!data->A0){
    // We store the initial ensemble to use it in final update equation                     (Line 11)
    bool dbg = ies_enkf_config_get_ies_debug( data->config ) ;
    int m_state_size = util_int_min(matrix_get_rows( A )-1, 50);
    int m_ens_size   = util_int_min(matrix_get_columns( A )-1, 16);
    fprintf(data->log_fp,"Allocating and assigning data->A0 \n");
    data->A0 = matrix_alloc_copy(A);
    if (dbg)
      matrix_pretty_fprint_submat(data->A0,"Ini data->A0","%11.5f",data->log_fp,0,m_state_size,0,m_ens_size);
  }
}

void ies_enkf_data_allocateW(ies_enkf_data_type * data, int ens_size) {
  if (!data->W){
    // We initialize data-W which will store W for use in next iteration                    (Line 9)
    bool dbg = ies_enkf_config_get_ies_debug( data->config ) ;
    int m_ens_size   = util_int_min(ens_size-1, 16);
    fprintf(data->log_fp,"Allocating data->W\n");
    data->W=matrix_alloc(ens_size , ens_size);
    matrix_set(data->W , 0.0) ;
    if (dbg)
      matrix_pretty_fprint_submat(data->W,"Ini data->W","%11.5f",data->log_fp,0,m_ens_size,0,m_ens_size) ;
  }
}

const bool_vector_type * ies_enkf_data_get_obs_mask0( const ies_enkf_data_type * data) {
  return data->obs_mask0;
}

const bool_vector_type * ies_enkf_data_get_obs_mask( const ies_enkf_data_type * data) {
  return data->obs_mask;
}

const bool_vector_type * ies_enkf_data_get_ens_mask( const ies_enkf_data_type * data) {
  return data->ens_mask;
}



const matrix_type * ies_enkf_data_getE(const ies_enkf_data_type * data) {
  return data->E;
}

matrix_type * ies_enkf_data_getW(const ies_enkf_data_type * data) {
  return data->W;
}

const matrix_type * ies_enkf_data_getA0(const ies_enkf_data_type * data) {
  return data->A0;
}
