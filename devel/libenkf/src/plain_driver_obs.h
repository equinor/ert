#ifndef __PLAIN_DRIVER_OBS_H__
#define __PLAIN_DRIVER_OBS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

typedef struct plain_driver_obs_struct plain_driver_obs_type;

plain_driver_obs_type       * plain_driver_obs_fread_alloc(const char * , FILE * );
  void 			    * plain_driver_obs_alloc(const char * , const char *);
void 			      plain_driver_obs_README(const char * );
void 			      plain_driver_obs_fwrite_mount_info(FILE * stream , const char * );

#ifdef __cplusplus
}
#endif
#endif
