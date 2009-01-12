#ifndef __PLAIN_DRIVER_DYNAMICH__
#define __PLAIN_DRIVER_DYNAMICH__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

typedef struct plain_driver_dynamic_struct plain_driver_dynamic_type;

plain_driver_dynamic_type   * plain_driver_dynamic_fread_alloc(const char * , FILE * );
void 			    * plain_driver_dynamic_alloc(const char * , const char * , const char *);
void 			      plain_driver_dynamic_fwrite_mount_info(FILE * stream , bool , const char *  , const char *);

#ifdef __cplusplus
}
#endif
#endif
