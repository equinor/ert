#ifndef __ENSEMBLE_CONFIG_H__
#define __ENSEMBLE_CONFIG_H__
#include <config.h>
#include <enkf_config_node.h>
#include <enkf_types.h>
#include <ecl_grid.h>
#include <stringlist.h>

typedef struct ensemble_config_struct ensemble_config_type;

void                     ensemble_config_add_obs_key(ensemble_config_type * , const char * , const char * );
const enkf_config_node_type  * ensemble_config_user_get_node(const ensemble_config_type *  , const char * , char **);
enkf_impl_type           ensemble_config_impl_type(const ensemble_config_type *, const char * );
enkf_var_type            ensemble_config_var_type(const ensemble_config_type *, const char * );
ensemble_config_type   * ensemble_config_alloc(const config_type * config , const ecl_grid_type *);
void                     ensemble_config_free(ensemble_config_type * );
bool                     ensemble_config_has_key(const ensemble_config_type * , const char * );
enkf_config_node_type  * ensemble_config_add_node(ensemble_config_type * ,
						const char    * ,
						enkf_var_type ,
						enkf_impl_type ,
						const char   * ,
						const char   * ,
						const void   * );

int                           ensemble_config_get_size(const ensemble_config_type * );
void                          ensemble_config_init_internalization( ensemble_config_type * );
void 			      ensemble_config_ensure_summary(ensemble_config_type *  , const char * );
void 			      ensemble_config_del_node(ensemble_config_type * , const char * );
void                          ensemble_config_add_config_items(config_type * );
enkf_config_node_type       * ensemble_config_get_node(const ensemble_config_type * , const char * );
stringlist_type             * ensemble_config_alloc_keylist(const ensemble_config_type *);
stringlist_type             * ensemble_config_alloc_keylist_from_var_type(const ensemble_config_type *  , int );
stringlist_type             * ensemble_config_alloc_keylist_from_impl_type(const ensemble_config_type *  , enkf_impl_type );
bool                          ensemble_config_iget_keep_runpath(const ensemble_config_type * , int );

#endif
