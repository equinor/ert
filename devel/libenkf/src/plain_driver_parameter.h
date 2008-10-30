#ifndef __PLAIN_DRIVER_PARAMETER_H__
#define __PLAIN_DRIVER_PARAMETER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <path_fmt.h>


typedef struct plain_driver_parameter_struct plain_driver_parameter_type;

plain_driver_parameter_type * plain_driver_parameter_fread_alloc(const char * , FILE * );
void                        * plain_driver_parameter_alloc(const char * , const char *);
void 			      plain_driver_parameter_fwrite_mount_info(FILE * stream , const char *);

#ifdef __cplusplus
}
#endif
#endif
