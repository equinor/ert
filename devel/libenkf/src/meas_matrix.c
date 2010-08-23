/**
   See the file README.obs for ducumentation of the varios datatypes
   involved with observations/measurement/+++.
*/

#include <math.h>
#include <stdlib.h>
#include <meas_matrix.h>
#include <string.h>
#include <util.h>
#include <hash.h>
#include <matrix.h>
#include <pthread.h>
#include <set.h>
#include <vector.h>

#define MEAS_BLOCK_TYPE_ID 661936407

struct meas_matrix_struct {
  int                 ens_size;
  vector_type       * data; 
  pthread_mutex_t     data_mutex;
  set_type          * lookup_keys;  /* Mangled obs_key and report_step */ 
};


struct meas_block_struct {
  UTIL_TYPE_ID_DECLARATION;
  int          ens_size;
  int          obs_size;
  int          ens_stride;
  int          obs_stride;
  double     * data;
  char       * obs_key;
  bool       * active;
};


static UTIL_SAFE_CAST_FUNCTION( meas_block , MEAS_BLOCK_TYPE_ID )


/**
   Observe that meas_block instance must be allocated with a correct
   value for obs_size; it can not grow during use, and it does also
   not count the number of elements added.

   Observe that the input argument @obs_size should be the total size
   of the observation; if parts of the observation have been excluded
   due to local analysis it should still be included in the @obs_size
   value.
*/

static meas_block_type * meas_block_alloc( const char * obs_key , int ens_size , int obs_size) {
  meas_block_type * meas_block = util_malloc( sizeof * meas_block , __func__ );
  UTIL_TYPE_ID_INIT( meas_block , MEAS_BLOCK_TYPE_ID );
  meas_block->ens_size    = ens_size;
  meas_block->obs_size    = obs_size;
  meas_block->obs_key     = util_alloc_string_copy( obs_key );
  meas_block->data        = util_malloc( (ens_size + 2)     * obs_size * sizeof * meas_block->data , __func__);
  meas_block->active      = util_malloc(  obs_size * sizeof * meas_block->active , __func__);
  meas_block->ens_stride  = 1;
  meas_block->obs_stride  = ens_size + 2; 
  {
    int i;
    for (i=0; i  <obs_size; i++)
      meas_block->active[i] = false;
  }
  return meas_block;
}

static void meas_block_free( meas_block_type * meas_block ) {
  free( meas_block->obs_key );
  free( meas_block->data );
  free( meas_block->active );
  free( meas_block );
}


static void meas_block_free__( void * arg ) {
  meas_block_type * meas_block = meas_block_safe_cast( arg );
  meas_block_free( meas_block );
}



static void meas_block_initS( const meas_block_type * meas_block , matrix_type * S, int * __obs_offset) {
  int obs_offset = *__obs_offset;
  for (int iobs =0; iobs < meas_block->obs_size; iobs++) {
    if (meas_block->active[iobs]) {
      for (int iens =0; iens < meas_block->ens_size; iens++) {
        int obs_index = iens * meas_block->ens_stride + iobs* meas_block->obs_stride;
        
        matrix_iset( S , obs_offset, iens , meas_block->data[ obs_index ]);
      }
      obs_offset++;
    }
  }
  *__obs_offset = obs_offset;
}



void meas_block_calculate_ens_stats( meas_block_type * meas_block ) {
  bool include_inactive = true;
  int iobs , iens;
  for (iobs =0; iobs < meas_block->obs_size; iobs++) {
    if (meas_block->active[iobs] || include_inactive) { 
      double M1 = 0;
      double M2 = 0;
      for (iens =0; iens < meas_block->ens_size; iens++) {
        int index = iens * meas_block->ens_stride + iobs * meas_block->obs_stride;
        M1 += meas_block->data[ index ];
        M2 += meas_block->data[ index ] * meas_block->data[ index ];
      }
      {
        int mean_index = (meas_block->ens_size + 0) * meas_block->ens_stride + iobs * meas_block->obs_stride;
        int std_index  = (meas_block->ens_size + 1) * meas_block->ens_stride + iobs * meas_block->obs_stride;
        double mean    = M1 / meas_block->ens_size; 
        double var     = M2 / meas_block->ens_size - mean * mean;
        meas_block->data[ mean_index ] = mean;
        meas_block->data[ std_index ]  = sqrt( util_double_max( 0.0 , var));
      }
    }
  }
}



void meas_block_iset( meas_block_type * meas_block , int iens , int iobs , double value) {
  int index = iens * meas_block->ens_stride + iobs * meas_block->obs_stride;
  meas_block->data[ index ] = value;
  if (!meas_block->active[ iobs ]) 
    meas_block->active[ iobs ] = true;

}


double meas_block_iget_ens_std( const meas_block_type * meas_block , int iobs) {
  int std_index  = (meas_block->ens_size + 1) * meas_block->ens_stride + iobs * meas_block->obs_stride;
  return meas_block->data[ std_index ];
}


double meas_block_iget_ens_mean( const meas_block_type * meas_block , int iobs) {
  int mean_index  = meas_block->ens_size * meas_block->ens_stride + iobs * meas_block->obs_stride;
  return meas_block->data[ mean_index ];
}


bool meas_block_iget_active( const meas_block_type * meas_block , int iobs) {
  return meas_block->active[ iobs ];
}


void meas_block_deactivate( meas_block_type * meas_block , int iobs ) {
  if (meas_block->active[ iobs ]) 
    meas_block->active[ iobs ] = false;
}


int meas_block_get_total_size( const meas_block_type * meas_block ) {
  return meas_block->obs_size;
}



/*****************************************************************/


meas_matrix_type * meas_matrix_alloc(int ens_size) {
  meas_matrix_type * meas = util_malloc(sizeof * meas , __func__);
  if (ens_size <= 0) 
    util_abort("%s: ens_size must be > 0 - aborting \n",__func__);

  meas->ens_size     = ens_size;
  meas->data         = vector_alloc_new();
  meas->lookup_keys  = set_alloc_empty();
  pthread_mutex_init( &meas->data_mutex , NULL );

  return meas;
}



void meas_matrix_free(meas_matrix_type * matrix) {
  vector_free( matrix->data );
  set_free( matrix->lookup_keys );
  free( matrix );
}



void meas_matrix_reset(meas_matrix_type * matrix) {
  set_clear( matrix->lookup_keys );
  vector_clear( matrix->data );
}


/**
   The code actually adding new blocks to the vector must be run in single-thread mode. 
*/

meas_block_type * meas_matrix_add_block( meas_matrix_type * matrix , const char * obs_key , int report_step , int obs_size) {
  char * lookup_key = util_alloc_sprintf( "%s-%d" , obs_key , report_step );  /* The obs_key is not alone unique over different report steps. */
  pthread_mutex_lock( &matrix->data_mutex );
  {
    if (!set_has_key( matrix->lookup_keys , lookup_key )) {
      meas_block_type  * new_block = meas_block_alloc(obs_key , matrix->ens_size , obs_size);
      vector_append_owned_ref( matrix->data , new_block , meas_block_free__ );
      set_add_key( matrix->lookup_keys , lookup_key );
    }
  }
  pthread_mutex_unlock( &matrix->data_mutex );
  free( lookup_key );
  return vector_get_last( matrix->data );
}



meas_block_type * meas_matrix_iget_block( meas_matrix_type * matrix , int block_nr) {
  return vector_iget( matrix->data , block_nr);
}


const meas_block_type * meas_matrix_iget_block_const( const meas_matrix_type * matrix , int block_nr) {
  return vector_iget_const( matrix->data , block_nr);
}



matrix_type * meas_matrix_allocS(const meas_matrix_type * matrix, int active_size) {
  int obs_offset = 0;
  matrix_type * S  = matrix_alloc( active_size , matrix->ens_size );

  for (int block_nr = 0; block_nr < vector_get_size( matrix->data ); block_nr++) {
    const meas_block_type * meas_block = vector_iget_const( matrix->data , block_nr);
    meas_block_initS( meas_block , S , &obs_offset);
  }
  return S;
}



int meas_matrix_get_nrobs( const meas_matrix_type * meas_matrix ) {
  return -1;
}


int meas_matrix_get_ens_size( const meas_matrix_type * meas_matrix ) {
  return meas_matrix->ens_size;
}
