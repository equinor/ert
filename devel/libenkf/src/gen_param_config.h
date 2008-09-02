#ifndef __GEN_PARAM_CONFIG_H__
#define __GEN_PARAM_CONFIG_H__
typedef struct gen_param_config_struct gen_param_config_type; 

int                     gen_param_config_get_data_size(const gen_param_config_type * );
gen_param_config_type * gen_param_config_alloc(const char * , const char * );
void             	gen_param_config_free(gen_param_config_type * );
void             	gen_param_config_assert_size(gen_param_config_type *  , int );
const bool     * 	gen_param_config_get_iactive(const gen_param_config_type * );
VOID_FREE_HEADER(gen_param_config)

#endif
