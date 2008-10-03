#ifndef __SITE_CONFIG_H__
#define __SITE_CONFIG_H__
#include <config.h>

typedef struct site_config_struct site_config_type;

site_config_type * site_config_alloc(const config_type * , int);
void               site_config_free(site_config_type *); 



#endif 
