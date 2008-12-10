#ifndef __ENKF_FS_H__
#define __ENKF_FS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <path_fmt.h>
#include <basic_driver.h>
#include <enkf_node.h>
#include <enkf_config_node.h>
#include <fs_index.h>

typedef struct enkf_fs_struct enkf_fs_type;

enkf_fs_type *   enkf_fs_mount(const char * , const char * , const char *);
enkf_fs_type * 	 enkf_fs_alloc(fs_index_type * , void * , void * , void *);
void           	 enkf_fs_free(enkf_fs_type *);
void           	 enkf_fs_add_index_node(enkf_fs_type *  , int , int , const char * , enkf_var_type, enkf_impl_type);
void 	       	 enkf_fs_fwrite_node(enkf_fs_type * , enkf_node_type * , int , int , state_enum );
void 	       	 enkf_fs_fread_node(enkf_fs_type  * , enkf_node_type * , int , int , state_enum );
bool 	       	 enkf_fs_has_node(enkf_fs_type  * , const enkf_config_node_type * , int , int , state_enum );
void           	 enkf_fs_fwrite_restart_kw_list(enkf_fs_type * , int , int , restart_kw_list_type *);
void           	 enkf_fs_fread_restart_kw_list(enkf_fs_type * , int , int , restart_kw_list_type *);
enkf_node_type * enkf_fs_fread_alloc_node(enkf_fs_type *  , enkf_config_node_type * , int  , int , state_enum );

#ifdef __cplusplus
}
#endif
#endif
