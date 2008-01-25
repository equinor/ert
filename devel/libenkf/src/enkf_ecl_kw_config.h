#ifndef __ENKF_ECL_KW_CONFIG_H__
#define __ENKF_ECL_KW_CONFIG_H__
#include <enkf_util.h>
#include <config.h>

typedef struct enkf_ecl_kw_config_struct enkf_ecl_kw_config_type;

struct enkf_ecl_kw_config_struct {
  CONFIG_STD_FIELDS;
};


enkf_ecl_kw_config_type * enkf_ecl_kw_config_alloc(int  , const char *, const char * );
enkf_ecl_kw_config_type * enkf_ecl_kw_config_alloc(int , const char * , const char *);
int        		  enkf_ecl_kw_config_get_size       (const enkf_ecl_kw_config_type *);
const char 		* enkf_ecl_kw_config_get_ensfile_ref(const enkf_ecl_kw_config_type *);
void                      enkf_ecl_kw_config_free(enkf_ecl_kw_config_type *);


VOID_FREE_HEADER(enkf_ecl_kw_config);
GET_SIZE_HEADER(enkf_ecl_kw);
VOID_GET_SIZE_HEADER(enkf_ecl_kw);

#endif
