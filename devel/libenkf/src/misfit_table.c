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
#include <buffer.h>

/**
   This file implements a type misfit_table which is used to rank the
   different realization according to various criteria.

   The top level datastructure in this file is the misfit_table, and
   that is the only exported datatype, but in addition there are the
   misfit_node which is the misfit for one ensemble member, and
   misfit_vector which is the misfit for one ensemble member / one
   observation key.
*/


typedef struct misfit_node_struct    misfit_node_type;
typedef struct misfit_vector_struct  misfit_vector_type;
typedef struct misfit_ranking_struct misfit_ranking_type;



#define MISFIT_TABLE_TYPE_ID  	    441066
#define MISFIT_NODE_TYPE_ID   	    541066
#define MISFIT_VECTOR_TYPE_ID 	    641066
#define MISFIT_RANKING_TYPE_ID      671108


/**
   This struct contains the misfits & sort keys for one particular
   ranking. I.e. all the RFT measurements.
*/


struct misfit_ranking_struct {
  UTIL_TYPE_ID_DECLARATION;
  vector_type        * ensemble;             /* An ensemble of hash instances. Each hash instance is populated like this: hash_insert_double(hash , "WGOR" , 1.09); */
  double_vector_type * total;                /* An enemble of total misfit values (for this ranking). */
  int                * sort_permutation;     /* This is how the ens members should be permuted to be sorted under this ranking.                                     */
};




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
  hash_type   *obs; 	      /* hash table of misfit_vector_type instances - indexed by observation keys. The structure
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

  if (history_length > 0)
    misfit_vector->data = double_vector_alloc( history_length + 1 , 0 );
  else
    misfit_vector->data = NULL;  /* Used by the xxx_fread_alloc() function below. */

  return misfit_vector;
}


static misfit_vector_type * misfit_vector_buffer_fread_alloc( buffer_type * buffer ) {
  misfit_vector_type * misfit_vector = misfit_vector_alloc( 0 );
  if (misfit_vector->data == NULL)
    misfit_vector->data = double_vector_buffer_fread_alloc( buffer);
  return misfit_vector;
}


static void misfit_vector_buffer_fwrite( const misfit_vector_type * misfit_vector , buffer_type * buffer ) {
  double_vector_buffer_fwrite( misfit_vector->data , buffer );
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


static void misfit_node_install_vector( misfit_node_type * node , const char * key , misfit_vector_type * vector ) {
  hash_insert_hash_owned_ref( node->obs, key , vector , misfit_vector_free__ );
}


static misfit_vector_type * misfit_node_safe_get_vector( misfit_node_type * node , const char * obs_key , int history_length) {
  if (!hash_has_key( node->obs , obs_key ))
    misfit_node_install_vector(node , obs_key , misfit_vector_alloc( history_length ) );
  return hash_get( node->obs , obs_key );
}


static misfit_vector_type * misfit_node_get_vector( const misfit_node_type * node , const char * obs_key ) {
  return hash_get( node->obs , obs_key );
}


static void misfit_node_update( misfit_node_type * node , const char * obs_key , int history_length , int iens , const double ** work_chi2) {
  misfit_vector_type * vector = misfit_node_safe_get_vector( node , obs_key , history_length );
  for (int step = 0; step <= history_length; step++) 
    misfit_vector_iset( vector , step , work_chi2[step][iens]);
}


static void misfit_node_buffer_fwrite( const misfit_node_type * node , buffer_type * buffer ) {
  buffer_fwrite_int( buffer , node->my_iens );
  buffer_fwrite_int( buffer , hash_get_size( node->obs ));
  {
    hash_iter_type * obs_iter = hash_iter_alloc( node->obs );
    while ( !hash_iter_is_complete( obs_iter )) {
      const char * key                   = hash_iter_get_next_key( obs_iter );
      misfit_vector_type * misfit_vector = hash_get( node->obs , key );
      buffer_fwrite_string( buffer , key );
      misfit_vector_buffer_fwrite( misfit_vector , buffer);
    }
    
    hash_iter_free( obs_iter );
  }
}


static misfit_node_type * misfit_node_buffer_fread_alloc( buffer_type * buffer ) {
  misfit_node_type * node;
  int my_iens   = buffer_fread_int( buffer );
  int hash_size = buffer_fread_int( buffer );
  node = misfit_node_alloc( my_iens );
  {
    int iobs;
    for (iobs = 0; iobs < hash_size; iobs++) {
      const char         * key           = buffer_fread_string( buffer );
      misfit_vector_type * misfit_vector = misfit_vector_buffer_fread_alloc( buffer );
      misfit_node_install_vector( node , key , misfit_vector );
    }
  }
  return node;
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
    obs_vector_ensemble_chi2( obs_vector , fs , 0 , misfit_table->history_length + 1 , 0 , ens_size , BOTH , chi2_work);
    
    
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


void misfit_table_buffer_fwrite( const misfit_table_type * misfit_table , buffer_type * buffer ) {
  int ens_size = vector_get_size( misfit_table->ensemble);
  buffer_fwrite_string(buffer , misfit_table->current_case );
  buffer_fwrite_int( buffer , misfit_table->history_length );
  buffer_fwrite_int( buffer , vector_get_size( misfit_table->ensemble ));

  /* Writing the nodes - one for each ensemble member */
  {
    int iens;
    for (iens = 0; iens < ens_size; iens++) 
      misfit_node_buffer_fwrite( vector_iget( misfit_table->ensemble , iens ) , buffer ); 
  }
  
  /* Does not store rankings currently */
}


/**
   Dumps a misfit table to file.
*/

void misfit_table_fwrite( const misfit_table_type * misfit_table , const char * filename ) {
  buffer_type * buffer = buffer_alloc( 1024 );
  misfit_table_buffer_fwrite( misfit_table , buffer );
  buffer_store( buffer , filename );
  buffer_free( buffer );
}


/**
   Observe that the object is NOT in a valid state when leaving this function, 
   must finalize in either misfit_table_alloc() or misfit_table_fread_alloc().
*/

static misfit_table_type * misfit_table_alloc_empty(const enkf_obs_type * enkf_obs) {
  misfit_table_type * table = util_malloc( sizeof * table , __func__);
  table->enkf_obs           = enkf_obs;
  table->ensemble           = vector_alloc_new();
  table->ranking_list       = hash_alloc();
  return table;
}



misfit_table_type * misfit_table_fread_alloc( const char * filename , const enkf_obs_type * enkf_obs) {
  misfit_table_type * misfit_table = misfit_table_alloc_empty( enkf_obs );
  buffer_type * buffer = buffer_fread_alloc( filename );
  int ens_size;
  
  misfit_table->current_case   = buffer_fread_alloc_string( buffer );
  misfit_table->history_length = buffer_fread_int( buffer );
  ens_size                     = buffer_fread_int( buffer );
  {
    for (int iens = 0; iens < ens_size; iens++) {
      misfit_node_type * node = misfit_node_buffer_fread_alloc( buffer );
      vector_append_owned_ref( misfit_table->ensemble , node , misfit_node_free__);
    }
  }

  buffer_free( buffer );
  return misfit_table;
}



misfit_table_type * misfit_table_alloc( const ensemble_config_type * config , enkf_fs_type * fs , int history_length , int ens_size , const enkf_obs_type * enkf_obs ) {
  misfit_table_type * table = misfit_table_alloc_empty( enkf_obs );
  table->current_case       = util_alloc_string_copy( enkf_fs_get_read_dir( fs ));
  table->history_length     = history_length;
  {
    int iens;
    for (iens = 0; iens < ens_size; iens++) 
      vector_append_owned_ref( table->ensemble , misfit_node_alloc( iens ) , misfit_node_free__);
  }
  misfit_table_update(table , config , fs);
  return table;
}



void misfit_table_display_ranking( const misfit_table_type * table , const char * ranking_key ) {
  const int ens_size                  = vector_get_size( table->ensemble );
  const misfit_ranking_type * ranking = hash_get( table->ranking_list , ranking_key );
  const int * permutations            = ranking->sort_permutation;
  int i;
  
  printf("\n\n");
  printf("  #    Realization    Total misfit\n");
  printf("-----------------------------------\n");
  for (i = 0; i < ens_size; i++) {
    int    iens         = permutations[i];
    double total_misfit = double_vector_iget( ranking->total , iens );
    printf("%3d    %3d               %10.3f  \n",i,iens,total_misfit);
  }
  printf("-----------------------------------\n");
}



void misfit_table_fprintf_ranking( const misfit_table_type * table , const char * ranking_key , const char * filename) {
  FILE * stream                       = util_mkdir_fopen( filename , "w");
  const int ens_size                  = vector_get_size( table->ensemble );
  const misfit_ranking_type * ranking = hash_get( table->ranking_list , ranking_key );
  const int * permutations            = ranking->sort_permutation;
  {
    const char * key_fmt       = "  %12s";
    const char * dash_line     = "--------------";
    const char * value_fmt     = " %12.3f ";
    const char * start_fmt     = " %3d       %3d           %10.3f      ";  /* <- This is significant whitespace ... */
    hash_type * obs_hash       = vector_iget( ranking->ensemble , 0);
    stringlist_type * obs_keys = hash_alloc_stringlist( obs_hash );
    int num_obs                = stringlist_get_size( obs_keys );
    int iobs;


    fprintf(stream , "   #     Realization   Total misfit     ");
    for (iobs =0; iobs < num_obs; iobs++) 
      fprintf(stream , key_fmt , stringlist_iget( obs_keys , iobs ));
    fprintf(stream , "\n");

    fprintf(stream , "----------------------------------------");
    for (iobs =0; iobs < num_obs; iobs++) 
      fprintf(stream , "%s" , dash_line);
    fprintf(stream , "\n");

    for (int i = 0; i < ens_size; i++) {
      int iens = permutations[i];
      hash_type * obs_hash = vector_iget( ranking->ensemble , iens );
      double total_value   = double_vector_iget( ranking->total , iens );
      fprintf(stream , start_fmt , i , iens , total_value);
      for (iobs =0; iobs < num_obs; iobs++) 
        fprintf(stream , value_fmt , hash_get_double( obs_hash , stringlist_iget( obs_keys , iobs )));
      fprintf(stream , "\n");
    }

    fprintf(stream , "----------------------------------------");
    for (iobs =0; iobs < num_obs; iobs++) 
      fprintf(stream , "%s" , dash_line);
    fprintf(stream , "\n");
    
  }
  fclose( stream );
}



const int * misfit_table_get_ranking_permutation( const misfit_table_type * table , const char * ranking_key ) {
  misfit_ranking_type * ranking = hash_get( table->ranking_list , ranking_key );
  return ranking->sort_permutation;
}


bool misfit_table_has_ranking( const misfit_table_type * table , const char * ranking_key ) {
  return hash_has_key( table->ranking_list , ranking_key );
}


static UTIL_SAFE_CAST_FUNCTION( misfit_ranking , MISFIT_RANKING_TYPE_ID )


static misfit_ranking_type * misfit_ranking_alloc( ) {
  misfit_ranking_type * misfit_ranking = util_malloc( sizeof * misfit_ranking , __func__);
  UTIL_TYPE_ID_INIT( misfit_ranking , MISFIT_RANKING_TYPE_ID );
  misfit_ranking->sort_permutation = NULL;
  misfit_ranking->ensemble = vector_alloc_new();
  misfit_ranking->total    = double_vector_alloc( 0 , 0 );
  return misfit_ranking;
}


static void misfit_ranking_free( misfit_ranking_type * misfit_ranking ) {
  vector_free( misfit_ranking->ensemble );
  double_vector_free( misfit_ranking->total );
  util_safe_free( misfit_ranking->sort_permutation );
  free( misfit_ranking );
}


static void misfit_ranking_free__( void * arg ) {
  misfit_ranking_type * ranking = misfit_ranking_safe_cast( arg );
  misfit_ranking_free( ranking );
}


/**
   Step and step2 are inclusive. The time direction is flattened.
*/
void misfit_table_create_ranking(misfit_table_type * table , const stringlist_type * sort_keys , int step1 , int step2, const char * ranking_key , const char * filename) {
  const int ens_size = vector_get_size( table->ensemble );
  int iens;
  misfit_ranking_type * ranking = misfit_ranking_alloc();

  for (iens = 0; iens < ens_size; iens++) {
    double total = 0;
    hash_type * obs_hash = hash_alloc();
    const misfit_node_type * misfit_node = vector_iget( table->ensemble , iens );  /* Lookup in the master ensemble. */
    for (int ikey = 0; ikey < stringlist_get_size( sort_keys ); ikey++) {
      const char * obs_key        = stringlist_iget( sort_keys , ikey );
      misfit_vector_type * vector = misfit_node_get_vector( misfit_node , obs_key );
      double value                = misfit_vector_eval( vector , step1 , step2 );  /* Sum up the misfit for this key - and these timesteps. */
      hash_insert_double( obs_hash , obs_key , value);
      total += value;
    }
    vector_append_owned_ref( ranking->ensemble , obs_hash , hash_free__);
    double_vector_append( ranking->total , total );
  }
  ranking->sort_permutation = double_vector_alloc_sort_perm( ranking->total );
  hash_insert_hash_owned_ref(table->ranking_list , ranking_key , ranking , misfit_ranking_free__);
  if (filename != NULL)
    misfit_table_fprintf_ranking( table , ranking_key , filename );
}




void misfit_table_free(misfit_table_type * table ) {
  hash_free( table->ranking_list );
  vector_free( table->ensemble );
  free( table->current_case );
  free( table );
}
