#ifndef __PLAIN_DRIVER_H__
#define __PLAIN_DRIVER_H__

#include <stdio.h>
#include <fs_types.h>
#include <stdbool.h>

typedef struct plain_driver_struct plain_driver_type;

void                 plain_driver_fwrite_mount_info(FILE * stream , fs_driver_type driver_type , bool read , const char * fmt);
plain_driver_type  * plain_driver_fread_alloc(const char * root_path , FILE * stream);

#endif
