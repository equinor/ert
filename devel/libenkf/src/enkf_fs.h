#ifndef __ENKF_FS_H__
#define __ENKF_FS_H__
#include <path_fmt.h>
#include <enkf_node.h>

typedef struct enkf_fs_struct enkf_fs_type;


enkf_fs_type * enkf_fs_alloc(void * , void * , void * , void *);
void           enkf_fs_free(enkf_fs_type *);
void           enkf_fs_swapin_node(enkf_fs_type * , enkf_node_type * , int , int , bool );
void           enkf_fs_swapout_node(enkf_fs_type * , enkf_node_type * , int , int , bool );

#endif
