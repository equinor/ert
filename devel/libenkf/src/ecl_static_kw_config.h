#ifndef __ECL_STATIC_KW_CONFIG_H__
#define __ECL_STATIC_KW_CONFIG_H__
#include <enkf_macros.h>

typedef struct ecl_static_kw_config_struct ecl_static_kw_config_type;



ecl_static_kw_config_type * ecl_static_kw_config_alloc(int , const char * , const char *);
int          		    ecl_static_kw_config_get_size       (const ecl_static_kw_config_type *);
const char   		  * ecl_static_kw_config_get_ensname_ref(const ecl_static_kw_config_type *);
void                        ecl_static_kw_config_free(ecl_static_kw_config_type *);


VOID_FREE_HEADER(ecl_static_kw_config);

#endif
