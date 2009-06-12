#ifndef __ENKF_FS_H__
#define __ENKF_FS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <path_fmt.h>
#include <basic_driver.h>
#include <enkf_node.h>
#include <enkf_config_node.h>
#include <stringlist.h>
#include <stdbool.h>
#include <stringlist.h>

typedef struct enkf_fs_struct enkf_fs_type;

bool              enkf_fs_rw_equal(const enkf_fs_type * );
stringlist_type * enkf_fs_alloc_dirlist(const enkf_fs_type * );
const char     *  enkf_fs_get_read_dir(const enkf_fs_type * );
const char     *  enkf_fs_get_write_dir(const enkf_fs_type * );
void              enkf_fs_interactive_select_directory(void * );
void 		  enkf_fs_add_dir(enkf_fs_type * , const char * );
bool 		  enkf_fs_has_dir(const enkf_fs_type * , const char * );
bool              enkf_fs_select_write_dir(enkf_fs_type * , const char * , bool);
void              enkf_fs_select_read_dir(enkf_fs_type * , const char * );
enkf_fs_type *    enkf_fs_mount(const char * , const char * , const char *);
void           	  enkf_fs_free(enkf_fs_type *);
void           	  enkf_fs_add_index_node(enkf_fs_type *  , int , int , const char * , enkf_var_type, enkf_impl_type);
void 	       	  enkf_fs_fwrite_node(enkf_fs_type * , enkf_node_type * , int , int , state_enum );
void 	       	  enkf_fs_fread_node(enkf_fs_type  * , enkf_node_type * , int , int , state_enum );
bool 	       	  enkf_fs_has_node(enkf_fs_type  * , const enkf_config_node_type * , int , int , state_enum );
void           	  enkf_fs_fwrite_restart_kw_list(enkf_fs_type * , int , int , const stringlist_type *);
void           	  enkf_fs_fread_restart_kw_list(enkf_fs_type * , int , int , stringlist_type *);
enkf_node_type  * enkf_fs_fread_alloc_node(enkf_fs_type *  , const enkf_config_node_type * , int  , int , state_enum );
enkf_node_type ** enkf_fs_fread_alloc_ensemble( enkf_fs_type * fs , const enkf_config_node_type * config_node , int report_step , int iens1 , int iens2 , state_enum state);
void              enkf_fs_copy_node(enkf_fs_type *, enkf_config_node_type *, int, int, state_enum, int, int, state_enum);
void              enkf_fs_copy_ensemble(enkf_fs_type * , enkf_config_node_type * , int , state_enum , int , state_enum , int , int );
void              enkf_fs_scatter_node(enkf_fs_type *, enkf_config_node_type *, int, int, state_enum, int, int);
bool              enkf_fs_try_fread_node(enkf_fs_type *  , enkf_node_type *  , int  , int  , state_enum );


#ifdef __cplusplus
}
#endif
#endif
