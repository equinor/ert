#ifndef __ENKF_CONFIG_H__
#define __ENKF_CONFIG_H__
#include <stdbool.h>
#include <enkf_config_node.h>
#include <enkf_types.h>
#include <obs_data.h>

#define LOG_INPUT    1
#define LOG_ENKF     2 
#define LOG_OUTPUT   4


/*****************************************************************/



typedef struct enkf_config_struct enkf_config_type;

const char      ** enkf_config_get_well_list_ref(const enkf_config_type * , int *);
bool               enkf_config_get_endian_swap(const enkf_config_type * );
enkf_config_type * enkf_config_alloc(const char * , const char * , const char * , const char * , const char * , bool);
enkf_impl_type     enkf_config_impl_type(const enkf_config_type *, const char * );
bool               enkf_config_has_key(const enkf_config_type * , const char *);
void               enkf_config_add_type(enkf_config_type * , const char * , enkf_var_type , enkf_impl_type , const void *);
void               enkf_config_add_type0(enkf_config_type * , const char * , int , enkf_var_type , enkf_impl_type );
void               enkf_config_add_well(enkf_config_type * , const char * , int , const char ** );
const enkf_config_node_type * enkf_config_get_ref(const enkf_config_type * , const char * );
void                          enkf_config_free(enkf_config_type * );



#endif
