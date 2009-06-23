#include <stdlib.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <vector.h>
#include <enkf_obs.h>
#include <double_vector.h>
#include <misfit_table.h>
#include <enkf_fs.h>

/**
   This file implements a type misfit_table which is used to rank the
   different realization according to various criteria.

   The top level datastructure in this file is the misfit_table, and
   that is the only exported datatype, but in addition there are the
   misfit_node which is the misfit for one ensemble member, and
   misfit_vector which is the misfit for one ensemble member / one
   observation key.
*/


typedef struct misfit_node_struct   misfit_node_type;
typedef struct misfit_vector_struct misfit_vector_type;


#define DEFAULT_RANKING_KEY   "default"

#define MISFIT_TABLE_TYPE_ID  441066
#define MISFIT_NODE_TYPE_ID   541066
#define MISFIT_VECTOR_TYPE_ID 641066



struct misfit_table_struct {
  UTIL_TYPE_ID_DECLARATION;
  int         	        history_length;  
  int         	      * ranking;        /* The currently active permutation vector for the content in ensemble - only a pointer to an element owned by the ranking_list hash table. */
  vector_type 	      * ensemble;       /* Vector of misfit_node_type instances - one for each ensemble member. */
  hash_type   	      * ranking_list;   /* A hash table of stored permutation vectors. */
  const enkf_obs_type * enkf_obs;       /* A pointer to the active enkf_obs instance - NOT owned by the misfit_table. */
};




struct misfit_node_struct {
  UTIL_TYPE_ID_DECLARATION;
  int          my_iens;
  double       sort_value;  /* misfit value used for sorting/ranking of this ensemble member. */
  hash_type   *obs; 	    /* hash table of misfit_vector_type instances - indexed by observation keys. The structure
		    	       of this hash table is duplicated for each ensemble member; that feels like a violation
		    	       of the DRY principle ... */
};




struct misfit_vector_struct {
  UTIL_TYPE_ID_DECLARATION;
  double_vector_type  * data;    /* A double vector of length 'history_length' with actual misfit values. */
};



/******************************************************************/
/* 
   Implementation of the misfit_vector type. Contains the full
   timeseries of misfit for one member/one observation key. The
   implementation is fully static.
*/

static misfit_vector_type * misfit_vector_alloc(int history_length) {
  misfit_vector_type * misfit_vector = util_malloc( sizeof * misfit_vector , __func__);
  UTIL_TYPE_ID_INIT(misfit_vector , MISFIT_VECTOR_TYPE_ID);
  misfit_vector->data = double_vector_alloc( history_length , 0 );
  return misfit_vector;
}

static UTIL_SAFE_CAST_FUNCTION(misfit_vector , MISFIT_VECTOR_TYPE_ID);

static void misfit_vector_free( misfit_vector_type * misfit_vector) {
  double_vector_free( misfit_vector->data );
  free( misfit_vector );
}


static void misfit_vector_free__( void * vector ) {
  misfit_vector_free( misfit_vector_safe_cast( vector ));
}


/**
   Will return the sum over the half-open interval [step1, step2).
*/

static double misfit_vector_sum( const misfit_vector_type * vector , int step1 , int step2 ) {
  double sum = 0;
  const double * data = double_vector_get_const_ptr( vector->data );
  for (int step = step1; step < step2; step++) 
    sum += data[step];
  return sum;
}


static void misfit_vector_iset( misfit_vector_type * vector , int time_index , double value ) {
  double_vector_iset( vector->data , time_index , value );
}

/*****************************************************************/


static misfit_node_type * misfit_node_alloc(int iens) {
  misfit_node_type * node = util_malloc( sizeof * node , __func__);
  UTIL_TYPE_ID_INIT( node , MISFIT_NODE_TYPE_ID);
  node->my_iens    = iens;
  node->sort_value = 0;
  node->obs        = hash_alloc();
  return node;
}


static UTIL_SAFE_CAST_FUNCTION(misfit_node , MISFIT_NODE_TYPE_ID);


static void misfit_node_free( misfit_node_type * node ) {
  hash_free( node->obs );
  free( node );
}


static void misfit_node_free__( void * node ) {
  misfit_node_free( misfit_node_safe_cast( node ));
}


static void misfit_node_update( misfit_node_type * node , enkf_fs_type * fs , const obs_vector_type * obs_vector) {
  
}

/*****************************************************************/


misfit_table_type * misfit_table_alloc( int history_length , int ens_size , const enkf_obs_type * enkf_obs ) {
  misfit_table_type * table = util_malloc( sizeof * table , __func__);
  table->history_length = history_length;
  table->ensemble       = vector_alloc_new();
  {
    int iens;
    for (iens = 0; iens < ens_size; iens++) 
      vector_append_owned_ref( table->ensemble , misfit_node_alloc( iens ) , misfit_node_free__);
  }
  table->ranking_list = hash_alloc();
  {
    int * ranking_list = util_malloc( ens_size * sizeof * ranking_list , __func__);
    hash_insert_hash_owned_ref( table->ranking_list , DEFAULT_RANKING_KEY , ranking_list , free);
  }
  table->ranking  = hash_get(table->ranking_list , DEFAULT_RANKING_KEY);
  table->enkf_obs = enkf_obs;
  return table;
}




void misfit_table_update(misfit_table_type * misfit_table , enkf_fs_type * fs) {
  const char      * obs_key;
  hash_iter_type * iter = enkf_obs_alloc_iter( misfit_table->enkf_obs );
  obs_key = hash_iter_get_next_key(iter);
  while (obs_key != NULL) {
    obs_vector_type * obs_vector = enkf_obs_get_vector( misfit_table->enkf_obs , obs_key );
    for (int iens = 0; iens < vector_get_size( misfit_table->ensemble); iens++) {
      misfit_node_type * node = vector_iget( misfit_table->ensemble , iens);
      misfit_node_update( node , fs , obs_vector );
    }
    obs_key = hash_iter_get_next_key(iter);
  }
  hash_iter_free(iter);
}



void misfit_table_free(misfit_table_type * table ) {
  hash_free( table->ranking_list );
  vector_free( table->ensemble );
  free( table );
}
