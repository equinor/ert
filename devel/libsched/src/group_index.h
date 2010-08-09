#ifndef __GROUP_INDEX_H__
#define __GROUP_INDEX_H__

#ifdef __cplusplus 
extern "C" {
#endif
#include <sched_types.h>


typedef struct group_index_struct group_index_type;

group_index_type              * group_index_alloc( const char * group_name , const char * variable , const void * state_ptr , sched_history_callback_ftype * func );
void                            group_index_free( group_index_type * group_index );
void                            group_index_free__( void * arg );
sched_history_callback_ftype *  group_index_get_callback( const group_index_type * group_index );
const void                   *  group_index_get_state__( const void * index );
const void                   *  group_index_get_state( const group_index_type * group_index );
const char                   * group_index_get_name( const group_index_type * group_index );
const char                   * group_index_get_variable( const group_index_type * group_index );



UTIL_IS_INSTANCE_HEADER( group_index );
UTIL_SAFE_CAST_HEADER_CONST( group_index );

#ifdef __cplusplus
}
#endif
#endif
