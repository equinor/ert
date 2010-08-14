#ifndef __ACTIVE_LIST_H__
#define __ACTIVE_LIST_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_types.h>


typedef struct active_list_struct active_list_type;

active_list_type * active_list_alloc( active_mode_type mode ); 
void               active_list_reset(active_list_type * );
void               active_list_add_index(active_list_type * , int);
void               active_list_free( active_list_type *);
const int        * active_list_get_active(const active_list_type * );
int                active_list_get_active_size(const active_list_type * , int total_size );
void               active_list_set_all_active(active_list_type * );
void               active_list_set_data_size(active_list_type *  , int );
void               active_list_free( active_list_type * );
active_mode_type   active_list_get_mode(const active_list_type * );
void               active_list_free__( void * arg );
active_list_type * active_list_alloc_copy( const active_list_type * src);
void               active_list_fprintf( const active_list_type * active_list , bool obs , const char * key , FILE * stream );

#ifdef __cplusplus
}
#endif
#endif
