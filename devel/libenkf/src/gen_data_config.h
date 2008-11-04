#ifndef __GEN_PARAM_CONFIG_H__
#define __GEN_PARAM_CONFIG_H__
#include <enkf_macros.h>
#include <stdbool.h>
typedef struct gen_param_config_struct gen_param_config_type;

char                  * gen_param_config_alloc_initfile(const gen_param_config_type *  , int );
ecl_type_enum           gen_param_config_get_ecl_type(const gen_param_config_type * );
int                     gen_param_config_get_byte_size(const gen_param_config_type * );
int                     gen_param_config_get_data_size(const gen_param_config_type * );
gen_param_config_type * gen_param_config_alloc(const char * , const char * );
void             	gen_param_config_free(gen_param_config_type * );
void             	gen_param_config_assert_size(gen_param_config_type *  , int , const char *);
const bool     * 	gen_param_config_get_iactive(const gen_param_config_type * );
void                    gen_param_config_ecl_write(const gen_param_config_type *  , const char * , char * );

GET_ACTIVE_SIZE_HEADER(gen_param)
GET_ACTIVE_LIST_HEADER(gen_param)
VOID_FREE_HEADER(gen_param_config)

#endif
