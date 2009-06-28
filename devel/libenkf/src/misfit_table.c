#include <stdlib.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <vector.h>
#include <enkf_obs.h>
#include <double_vector.h>
#include <misfit_table.h>
#include <enkf_fs.h>
#include <msg.h>

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


#define MISFIT_TABLE_TYPE_ID  	    441066
#define MISFIT_NODE_TYPE_ID   	    541066
#define MISFIT_VECTOR_TYPE_ID 	    641066





struct misfit_table_struct {
  UTIL_TYPE_ID_DECLARATION;
  int         	        history_length;  
  vector_type 	      * ensemble;       /* Vector of misfit_node_type instances - one for each ensemble member. */
  hash_type   	      * ranking_list;   /* A hash table of stored permutation vectors. */
  const enkf_obs_type * enkf_obs;       /* A pointer to the active enkf_obs instance - NOT owned by the misfit_table. */
  char                * current_case;   /* The (filesystem) case which was used when building the misfit table. */
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
  misfit_vector->data = double_vector_alloc( history_length + 1 , 0 );
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

/** Step2 is inclusive - what a fucking mess. */
static double misfit_vector_eval( const misfit_vector_type * vector , int step1 , int step2 ) {
  double misfit_sum = 0;
  int step;
  
  for (step = step1; step <= step2; step++)
    misfit_sum += double_vector_iget(vector->data , step );
  
  return misfit_sum;
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


static misfit_vector_type * misfit_node_safe_get_vector( misfit_node_type * node , const char * obs_key , int history_length) {
  if (!hash_has_key( node->obs , obs_key ))
    hash_insert_hash_owned_ref( node->obs, obs_key , misfit_vector_alloc( history_length ) , misfit_vector_free__);
  return hash_get( node->obs , obs_key );
}

static misfit_vector_type * misfit_node_get_vector( misfit_node_type * node , const char * obs_key ) {
  return hash_get( node->obs , obs_key );
}


static void misfit_node_update( misfit_node_type * node , const char * obs_key , int history_length , int iens , const double ** work_chi2) {
  misfit_vector_type * vector = misfit_node_safe_get_vector( node , obs_key , history_length );
  for (int step = 0; step <= history_length; step++) 
    misfit_vector_iset( vector , step , work_chi2[step][iens]);
}


static void misfit_node_eval_ranking_misfit( misfit_node_type * node , const stringlist_type * sort_keys , int step1 , int step2) {
  int ikey;
  node->sort_value = 0;

  for (ikey = 0; ikey < stringlist_get_size( sort_keys ); ikey++) {
    const char * obs_key        = stringlist_iget( sort_keys , ikey );
    misfit_vector_type * vector = misfit_node_get_vector( node , obs_key );
    node->sort_value += misfit_vector_eval( vector , step1 , step2 );
  }
}


/*****************************************************************/


static double ** __2d_malloc(int rows , int columns) {
  double ** d = util_malloc( rows * sizeof * d , __func__);
  for (int i =0; i < rows; i++)
    d[i] = util_malloc( columns * sizeof * d[i] , __func__);
  return d;
}

static void  __2d_free(double ** d , int rows) {
  for (int i =0; i < rows; i++)
    free(d[i]);
  free(d);
}



static void misfit_table_update( misfit_table_type * misfit_table , const ensemble_config_type * config , enkf_fs_type * fs ) {
  msg_type * msg            = msg_alloc("Evaluating misfit for observation: ");
  const int ens_size 	    = vector_get_size( misfit_table->ensemble );
  double ** chi2_work       = __2d_malloc( misfit_table->history_length + 1 , ens_size );
  hash_iter_type * obs_iter = enkf_obs_alloc_iter( misfit_table->enkf_obs );
  const char * obs_key      = hash_iter_get_next_key( obs_iter );

  msg_show( msg );
  while (obs_key != NULL) {
    obs_vector_type * obs_vector = enkf_obs_get_vector( misfit_table->enkf_obs , obs_key );
    
    msg_update( msg , obs_key );
    obs_vector_ensemble_chi2( obs_vector , fs , 0 , misfit_table->history_length + 1 , 0 , ens_size , both , chi2_work);
    
    
    /** 
	Internalizing the results from the chi2_work table into the misfit structure.
    */
    for (int iens = 0; iens < ens_size; iens++) {
      misfit_node_type * node = vector_iget( misfit_table->ensemble , iens);
      misfit_node_update( node , obs_key , misfit_table->history_length , iens , (const double **) chi2_work);
    }

    obs_key = hash_iter_get_next_key( obs_iter );
  }
  

  msg_free(msg , true );
  hash_iter_free( obs_iter );
  __2d_free( chi2_work , misfit_table->history_length + 1);
}




misfit_table_type * misfit_table_alloc( const ensemble_config_type * config , enkf_fs_type * fs , int history_length , int ens_size , const enkf_obs_type * enkf_obs ) {
  misfit_table_type * table = util_malloc( sizeof * table , __func__);
  table->current_case   = util_alloc_string_copy( enkf_fs_get_read_dir( fs ));
  table->history_length = history_length;
  table->ensemble       = vector_alloc_new();
  {
    int iens;
    for (iens = 0; iens < ens_size; iens++) 
      vector_append_owned_ref( table->ensemble , misfit_node_alloc( iens ) , misfit_node_free__);
  }
  table->ranking_list = hash_alloc();
  table->enkf_obs = enkf_obs;
  
  misfit_table_update(table , config , fs);
  return table;
}


void misfit_table_display_ranking( const misfit_table_type * table , const char * ranking_key ) {
  const int ens_size  = vector_get_size( table->ensemble );
  const int * ranking = hash_get( table->ranking_list , ranking_key );
  int i;
  
  printf("\n\n");
  printf("  #    Realization    Total misfit\n");
  printf("-----------------------------------\n");
  for (i = 0; i < ens_size; i++) {
    int iens = ranking[i];
    misfit_node_type * node = vector_iget( table->ensemble , iens );
    printf("%3d    %3d               %10.3f  \n",i,iens,node->sort_value);
  }
  printf("-----------------------------------\n");
}



const int * misfit_table_get_ranking_permutation( const misfit_table_type * table , const char * ranking_key ) {
  return hash_get( table->ranking_list , ranking_key );
}


bool misfit_table_has_ranking( const misfit_table_type * table , const char * ranking_key ) {
  return hash_has_key( table->ranking_list , ranking_key );
}




/**
   Step and step2 are inclusive.
*/
void misfit_table_create_ranking(misfit_table_type * table , const stringlist_type * sort_keys , int step1 , int step2, const char * ranking_key) {
  const int ens_size = vector_get_size( table->ensemble );
  int iens;
  for (iens = 0; iens < ens_size; iens++) {
    misfit_node_type * node = vector_iget( table->ensemble , iens );
    misfit_node_eval_ranking_misfit( node , sort_keys , step1 , step2);
  }
  {
    double_vector_type * rank_values = double_vector_alloc( ens_size , 0 );
    for (iens = 0; iens < ens_size; iens++) {
      misfit_node_type * node = vector_iget( table->ensemble , iens );
      double_vector_iset( rank_values , iens , node->sort_value );
    }
    hash_insert_hash_owned_ref(table->ranking_list , ranking_key , double_vector_alloc_sort_perm( rank_values ) , free);
    double_vector_free( rank_values );
  }
}




void misfit_table_free(misfit_table_type * table ) {
  hash_free( table->ranking_list );
  vector_free( table->ensemble );
  free( table->current_case );
  free( table );
}
