#include <stdlib.h>
#include <size_t_vector.h>
#include <int_vector.h>
#include <util.h>  
#include <sched_types.h>
#include <group_index.h>


#define GROUP_INDEX_TYPE_ID 96580631


struct group_index_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                         * group_name;
  char                         * variable;
  const void                   * group_history;
  sched_history_callback_ftype * func;                 
};





UTIL_IS_INSTANCE_FUNCTION( group_index , GROUP_INDEX_TYPE_ID )
UTIL_SAFE_CAST_FUNCTION_CONST( group_index , GROUP_INDEX_TYPE_ID )


group_index_type * group_index_alloc( const char * group_name , const char * variable , const void * group_history , sched_history_callback_ftype * func ) {
  group_index_type * group_index = util_malloc( sizeof * group_index , __func__ );
  
  UTIL_TYPE_ID_INIT( group_index , GROUP_INDEX_TYPE_ID );
  
  group_index->func      = func;
  group_index->group_history = group_history;
  group_index->group_name    = util_alloc_string_copy( group_name );
  group_index->variable      = util_alloc_string_copy( variable );

  
  return group_index;
}


void group_index_free( group_index_type * index ) {
  free( index->group_name );
  free( index->variable );
  free( index );
}


void group_index_free__( void * arg ) {
  group_index_free(  (group_index_type  *) arg );
}



sched_history_callback_ftype * group_index_get_callback( const group_index_type * group_index ) {
  return group_index->func;
}


const char * group_index_get_name( const group_index_type * group_index ) {
  return group_index->group_name;
}

const char * group_index_get_variable( const group_index_type * group_index ) {
  return group_index->variable;
}



const void * group_index_get_state( const group_index_type * group_index ) {
  return group_index->group_history;
}



const void * group_index_get_state__( const void * index ) {
  const group_index_type * group_index = group_index_safe_cast_const( index );
  return group_index_get_state( group_index );
}


