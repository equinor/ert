#ifndef __BLOCK_FS_DRIVER_H__
#define __BLOCK_FS_DRIVER_H__


#include <stdio.h>
#include <fs_types.h>
#include <stdbool.h>

typedef struct block_fs_driver_struct block_fs_driver_type;


void                   block_fs_driver_fwrite_mount_info(FILE * stream , fs_driver_enum driver_type , int num_block_fs_drivers);
block_fs_driver_type * block_fs_driver_fread_alloc(const char * root_path , FILE * stream);
bool                   block_fs_sscanf_key(const char * key , char ** config_key , int * __report_step , int * __iens);




#endif
