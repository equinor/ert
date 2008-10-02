#ifndef __ENSEMBLE_CONFIG_H__
#define __ENSEMBLE_CONFIG_H__
#include <config.h>
#include <enkf_config_node.h>


typedef struct ensemble_config_struct ensemble_config_type;


ensemble_config_type * ensemble_config_alloc_empty(int );
ensemble_config_type * ensemble_config_alloc(const config_type * config);
void                   ensemble_config_free(ensemble_config_type * );
bool                   ensemble_config_has_key(const ensemble_config_type * , const char * );
void                   ensemble_config_add_node(ensemble_config_type * ,
						const char    * ,
						enkf_var_type ,
						enkf_impl_type ,
						const char   * ,
						const char   * ,
						const void   * );

int                    ensemble_config_get_size(const ensemble_config_type * );


void                          ensemble_config_add_config_items(config_type * );
const enkf_config_node_type * ensemble_config_get_node(const ensemble_config_type * , const char * );

#endif
