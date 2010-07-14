#include <stdlib.h>
#include <size_t_vector.h>
#include <int_vector.h>
#include <util.h>  
#include <sched_types.h>
#include <well_index.h>


#define WELL_INDEX_TYPE_ID 99780634

struct well_index_struct {
  UTIL_TYPE_ID_DECLARATION;
  const void * state_ptr;
  int_vector_type         * kw_type;
  size_t_vector_type      * func;                 
};





UTIL_IS_INSTANCE_FUNCTION( well_index , WELL_INDEX_TYPE_ID )
UTIL_SAFE_CAST_FUNCTION_CONST( well_index , WELL_INDEX_TYPE_ID )


void well_index_add_type( well_index_type * index , sched_kw_type_enum kw_type , sched_history_callback_ftype * func) {
  int_vector_append( index->kw_type      , kw_type );
  size_t_vector_append( index->func      , ( size_t ) func );
}



well_index_type * well_index_alloc( const void * state_ptr , sched_kw_type_enum kw_type , sched_history_callback_ftype * func ) {
  well_index_type * well_index = util_malloc( sizeof * well_index , __func__ );
  
  UTIL_TYPE_ID_INIT( well_index , WELL_INDEX_TYPE_ID );
  
  
  well_index->kw_type   = int_vector_alloc( 0 , 0 );
  well_index->func      = size_t_vector_alloc( 0 , 0 );
  well_index->state_ptr = state_ptr;
  
  well_index_add_type( well_index , kw_type , func );
  return well_index;
}


void well_index_free( well_index_type * index ) {
  size_t_vector_free( index->func );
  int_vector_free( index->kw_type );
  free( index );
}


void well_index_free__( void * arg ) {
  well_index_free(  (well_index_type  *) arg );
}



sched_history_callback_ftype * well_index_get_callback( const well_index_type * well_index , sched_kw_type_enum kw_type) {
  sched_history_callback_ftype * func = NULL;
  int iindex = 0;
  while (true) {
    printf("Comparing %d %d \n",kw_type , int_vector_iget( well_index->kw_type , iindex));
    if (int_vector_iget( well_index->kw_type , iindex) == kw_type) {
      func = ( sched_history_callback_ftype *) size_t_vector_iget( well_index->func , iindex );
      break;
    }
    
    
    iindex++;
    if (iindex == int_vector_size( well_index->kw_type ))
      break;
  }
  
  return func;
}



const void * well_index_get_state( const well_index_type * well_index ) {
  return well_index->state_ptr;
}



const void * well_index_get_state__( const void * index ) {
  const well_index_type * well_index = well_index_safe_cast_const( index );
  return well_index_get_state( well_index );
}
