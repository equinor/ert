#ifndef __GEN_KW_CONFIG_H__
#define __GEN_KW_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <gen_kw_common.h>
#include <stringlist.h>
#include <util.h>


void                        gen_kw_config_set_parameter_file( gen_kw_config_type * config , const char * parameter_file );
const char                * gen_kw_config_get_parameter_file( const gen_kw_config_type * config );
char                      * gen_kw_config_alloc_initfile( const gen_kw_config_type * gen_kw_config , int iens );
void                        gen_kw_config_set_init_file_fmt( gen_kw_config_type * config , const char * init_file_fmt );
const char                * gen_kw_config_get_key(const gen_kw_config_type * config );
const char         	  * gen_kw_config_get_template_file(const gen_kw_config_type * );
void               	    gen_kw_config_free(gen_kw_config_type *);
double               	    gen_kw_config_transform(const gen_kw_config_type * , int index, double x);
int                	    gen_kw_config_get_data_size(const gen_kw_config_type * );
const char         	  * gen_kw_config_iget_name(const gen_kw_config_type * , int );
const char         	  * gen_kw_config_get_tagged_name(const gen_kw_config_type * , int );
stringlist_type           * gen_kw_config_alloc_name_list( const gen_kw_config_type * config);
int                         gen_kw_config_get_index(const gen_kw_config_type *  , const char * );
char                      * gen_kw_config_alloc_user_key(const gen_kw_config_type * config , int kw_nr);
const char                * gen_kw_config_get_init_file_fmt( const gen_kw_config_type * config );
void                        gen_kw_config_set_template_file( gen_kw_config_type * config , const char * template_file );
gen_kw_config_type        * gen_kw_config_alloc_empty(const char * key );
void                        gen_kw_config_update( gen_kw_config_type * config , const char * template_file , const char * parameter_file , const char * init_file_fmt);

UTIL_SAFE_CAST_HEADER_CONST( gen_kw_config );
UTIL_SAFE_CAST_HEADER(gen_kw_config);
VOID_FREE_HEADER(gen_kw_config);
VOID_GET_DATA_SIZE_HEADER(gen_kw);
#ifdef __cplusplus
}
#endif
#endif
