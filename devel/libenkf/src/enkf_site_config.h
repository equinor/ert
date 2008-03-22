#ifndef __ENKF_SITE_CONFIG_H__
#define __ENKF_SITE_CONFIG_H__

typedef struct enkf_site_config_struct       enkf_site_config_type;
typedef struct enkf_site_config_node_struct  enkf_site_config_node_type; 

enkf_site_config_type * enkf_site_config_bootstrap(const char * );
void                    enkf_site_config_free(enkf_site_config_type * );
void                    enkf_site_config_validate(enkf_site_config_type *);
bool                    enkf_site_config_node_set(const enkf_site_config_type * , const char * );
#endif
