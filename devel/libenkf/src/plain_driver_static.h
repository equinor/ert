#ifndef __PLAIN_DRIVER_STATIC_H__
#define __PLAIN_DRIVER_STATIC_H__
#include <path_fmt.h>


typedef struct plain_driver_static_struct plain_driver_static_type;

plain_driver_static_type * plain_driver_static_fread_alloc(const char * , FILE * );
void                     * plain_driver_static_alloc(const char * , const char *);
void                       plain_driver_static_fwrite_mount_info(FILE * stream , const char *);

#endif
