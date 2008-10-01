#ifndef __ECL_CONFIG_H__
#define __ECL_CONFIG_H__
#include <config.h>
#include <time.h>


typedef struct ecl_config_struct ecl_config_type;

ecl_config_type * ecl_config_alloc( const config_type *  , time_t * );
void              ecl_config_free( ecl_config_type *);
bool              ecl_config_include_static_kw(const ecl_config_type * , const char * );
void              ecl_config_add_static_kw(ecl_config_type *, const char *); 

#endif
