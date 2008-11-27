#ifndef __GEN_KW_CONFIG_H__
#define __GEN_KW_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <scalar_config.h>
#include <scalar.h>

typedef struct gen_kw_config_struct gen_kw_config_type;



const char         	  * gen_kw_config_get_template_ref(const gen_kw_config_type * );
gen_kw_config_type 	  * gen_kw_config_fscanf_alloc(const char * , const char *);
void               	    gen_kw_config_free(gen_kw_config_type *);
void               	    gen_kw_config_transform(const gen_kw_config_type * , const double * , double *);
void               	    gen_kw_config_truncate(const gen_kw_config_type * , scalar_type * );
int                	    gen_kw_config_get_data_size(const gen_kw_config_type * );
const char         	  * gen_kw_config_get_name(const gen_kw_config_type * , int );
const char         	  * gen_kw_config_get_tagged_name(const gen_kw_config_type * , int );
char               	 ** gen_kw_config_get_name_list(const gen_kw_config_type *);
const scalar_config_type  * gen_kw_config_get_scalar_config(const gen_kw_config_type * );
int                         gen_kw_config_get_index(const gen_kw_config_type *  , const char * );

VOID_FREE_HEADER(gen_kw_config);
#ifdef __cplusplus
}
#endif
#endif
