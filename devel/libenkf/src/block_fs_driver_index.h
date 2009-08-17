#ifndef __BLOCK_FS_DRIVER_INDEX__
#define __BLOCK_FS_DRIVER_INDEX__


#include <stdio.h>
#include <fs_types.h>
#include <stdbool.h>

typedef struct block_fs_driver_index_struct block_fs_driver_index_type;


void                         block_fs_driver_index_fwrite_mount_info(FILE * stream );
block_fs_driver_index_type * block_fs_driver_index_fread_alloc(const char * root_path , FILE * stream);






#endif
