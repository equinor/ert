#ifndef __PLAIN_DRIVER_INDEX_H__
#define __PLAIN_DRIVER_INDEX_H__
#include <stdio.h>

typedef struct plain_driver_index_struct plain_driver_index_type;

void * plain_driver_index_fread_alloc(const char *  , FILE * );
void   plain_driver_index_fwrite_mount_info(FILE *  , bool , const char * );

#endif
