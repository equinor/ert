#ifndef __MULTFLT_CONFIG_H__
#define __MULTFLT_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <scalar_config.h>
#include <scalar.h>

typedef struct multflt_config_struct multflt_config_type;


scalar_config_type  * multflt_config_get_scalar_config( const multflt_config_type * );
int                   multflt_config_get_index(const multflt_config_type *  , const char * );
multflt_config_type * multflt_config_fscanf_alloc(const char *);
void                  multflt_config_free(multflt_config_type *);
void                  multflt_config_transform(const multflt_config_type * , const double * , double *);
void                  multflt_config_truncate(const multflt_config_type * , scalar_type * );
int                   multflt_config_get_data_size(const multflt_config_type * );
const char          * multflt_config_get_name(const multflt_config_type * , int );
const char         ** multflt_config_get_names(const multflt_config_type *);

VOID_FREE_HEADER(multflt_config);
VOID_CONFIG_ACTIVATE_HEADER(multflt)
#ifdef __cplusplus
}
#endif
#endif
