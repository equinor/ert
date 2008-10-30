#ifndef __FS_INDEX_H__
#define __FS_INDEX_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_types.h>
#include <restart_kw_list.h>

typedef struct fs_index_struct fs_index_type; 

fs_index_type * fs_index_fread_alloc(const char *  , FILE * );
void            fs_index_fwrite_mount_info(FILE * , const char * );
fs_index_type * fs_index_alloc(const char * , const char *);
void            fs_index_free(fs_index_type *);
void            fs_index_add_node(fs_index_type *, int , int , const char *, enkf_var_type , enkf_impl_type );
bool            fs_index_has_node(fs_index_type *, int , int , const char *);
void            fs_index_fwrite_restart_kw_list(fs_index_type * , int , int , restart_kw_list_type * );
void            fs_index_fread_restart_kw_list(fs_index_type * , int , int , restart_kw_list_type * );

#ifdef __cplusplus
}
#endif
#endif
