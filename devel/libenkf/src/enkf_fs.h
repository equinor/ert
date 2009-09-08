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
#include <fs_types.h>

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
enkf_fs_type *    enkf_fs_mount(const char * , fs_driver_impl , const char * );
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
void              enkf_fs_copy_ensemble(enkf_fs_type * , enkf_config_node_type * , int , state_enum , int , state_enum , int , const int * permutations);
void              enkf_fs_scatter_node(enkf_fs_type *, enkf_config_node_type *, int, int, state_enum, int, int);
bool              enkf_fs_try_fread_node(enkf_fs_type *  , enkf_node_type *  , int  , int  , state_enum );


char             * enkf_fs_alloc_case_filename( const enkf_fs_type * fs , const char * input_name);
char             * enkf_fs_alloc_case_member_filename( const enkf_fs_type * fs , int iens , const char * input_name);
char             * enkf_fs_alloc_case_tstep_filename( const enkf_fs_type * fs , int tstep , const char * input_name);
char             * enkf_fs_alloc_case_tstep_member_filename( const enkf_fs_type * fs , int tstep , int iens , const char * input_name);
FILE             * enkf_fs_open_case_tstep_member_file( const enkf_fs_type * fs , const char * input_name , int tstep , int iens , const char * mode);
FILE             * enkf_fs_open_case_file( const enkf_fs_type * fs , const char * input_name , const char * mode);
FILE             * enkf_fs_open_case_tstep_file( const enkf_fs_type * fs , const char * input_name , int tstep , const char * mode);
FILE             * enkf_fs_open_case_member_file( const enkf_fs_type * fs , const char * input_name , int iens , const char * mode);


UTIL_SAFE_CAST_HEADER( enkf_fs );
UTIL_IS_INSTANCE_HEADER( enkf_fs );


#ifdef __cplusplus
}
#endif
#endif
