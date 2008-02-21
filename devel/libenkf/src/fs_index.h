#ifndef __FS_INDEX_H__
#define __FS_INDEX_H__
#include <enkf_types.h>

typedef struct fs_index_struct fs_index_type; 

bool            fs_index_has_node(fs_index_type *, int , const char *);
fs_index_type * fs_index_alloc(const char * , const char *);
void            fs_index_free(fs_index_type *);
void            fs_index_add_node(fs_index_type *, int , const char *, enkf_var_type , enkf_impl_type );


#endif
