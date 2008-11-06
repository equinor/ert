#ifndef __EQUIL_CONFIG_H__
#define __EQUIL_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <scalar_config.h>

typedef struct equil_config_struct equil_config_type;


const scalar_config_type * equil_config_get_scalar_config(const equil_config_type * );
void                	   equil_config_ecl_write(const equil_config_type *   , const double *  , const double * , FILE * );
int                 	   equil_config_get_nequil(const equil_config_type * );
equil_config_type   	 * equil_config_alloc(int);
void                	   equil_config_free(equil_config_type *);
int                 	   equil_config_get_nequil(const equil_config_type *);
equil_config_type   	 * equil_config_fscanf_alloc(const char * );

VOID_FREE_HEADER(equil_config);
VOID_CONFIG_ACTIVATE_HEADER(equil);
#ifdef __cplusplus
}
#endif
#endif
