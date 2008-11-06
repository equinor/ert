#ifndef __ACTIVE_LIST_H__
#define __ACTIVE_LIST_H__

#ifdef __cplusplus
extern "C" {
#endif



typedef struct active_list_struct active_list_type;

active_list_type * active_list_alloc( int ); 
void               active_list_reset(active_list_type * );
void               active_list_add_index(active_list_type * , int);
void               active_list_free( active_list_type *);
const int        * active_list_get_active(const active_list_type * );
int                active_list_get_active_size(const active_list_type * );
void               active_list_set_all_active(active_list_type * );
void               active_list_set_data_size(active_list_type *  , int );
void               active_list_free( active_list_type * );
void               active_list_grow(active_list_type *  , int );

#ifdef __cplusplus
}
#endif
#endif
