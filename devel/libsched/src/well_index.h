#ifndef __WELL_INDEX_H__
#define __WELL_INDEX_H__

#ifdef __cplusplus 
extern "C" {
#endif
#include <sched_types.h>


typedef struct well_index_struct well_index_type;


well_index_type              * well_index_alloc( const void * state_ptr , sched_kw_type_enum kw_type , sched_history_callback_ftype * func);
void                           well_index_free( well_index_type * well_index );
void                           well_index_free__( void * arg );
void                           well_index_add_type( well_index_type * index , sched_kw_type_enum kw_type , sched_history_callback_ftype * func);
sched_history_callback_ftype * well_index_get_callback( const well_index_type * well_index , sched_kw_type_enum kw_type);
const void                   * well_index_get_state__( const void * index );
const void                   * well_index_get_state( const well_index_type * well_index );

UTIL_IS_INSTANCE_HEADER( well_index );


#ifdef __cplusplus
}
#endif
#endif
