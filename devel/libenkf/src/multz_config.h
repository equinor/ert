#ifndef __MULTZ_CONFIG_H__
#define __MULTZ_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <scalar.h>
#include <scalar_config.h>

typedef struct {
  int                   __type_id;
  scalar_config_type  * scalar_config;
  int *i1, *i2 , *j1 , *j2 , *k;
  int 	           * area;
} multz_config_type;



double              multz_config_truncate(const multz_config_type * , int , double );
multz_config_type * multz_config_fscanf_alloc(const char * , int , int , int);
multz_config_type * multz_config_alloc(int , int , int , const char * , const char * );
void                multz_config_free(multz_config_type *); 
const        char * multz_config_get_ensfile_ref(const multz_config_type * );
const        char * multz_config_get_eclfile_ref(const multz_config_type * );
void                multz_config_ecl_write(const multz_config_type * , const double *, FILE *);
char              * multz_config_alloc_description(const multz_config_type * , int );


/*Generated headers */
GET_DATA_SIZE_HEADER(multz);
VOID_FREE_HEADER(multz_config);
VOID_CONFIG_ACTIVATE_HEADER(multz)
#ifdef __cplusplus
}
#endif
#endif
